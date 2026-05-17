#include <iostream>
#include <memory>
#include <string>

#include <drake/systems/primitives/constant_value_source.h>
#include <drake/systems/primitives/constant_vector_source.h>
#include <drake/systems/primitives/zero_order_hold.h>
#include <gflags/gflags.h>

#include "core/c3.h"
#include "examples/common_systems.hpp"
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/lcs_factory_system.h"
#include "systems/lcs_simulator.h"

#include "drake/common/proto/call_python.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/meshcat/contact_visualizer.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/vector_log_sink.h"

struct ConveyorSystem {
  std::unique_ptr<drake::systems::DiagramBuilder<double>> builder;
  std::unique_ptr<drake::systems::Diagram<double>> diagram;
  drake::multibody::MultibodyPlant<double>* plant{};
  drake::geometry::SceneGraph<double>* scene_graph{};
};

class SineVectorGenerator : public drake::systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SineVectorGenerator);
  SineVectorGenerator() {
    this->DeclareVectorOutputPort("sine_cosine",
                                  drake::systems::BasicVector<double>(6),
                                  &SineVectorGenerator::calc_output);
  }

  void calc_output(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output_vector) const {
    Eigen::VectorBlock<Eigen::VectorX<double>> output_value =
        output_vector->get_mutable_value();
    Eigen::VectorX<double> out = Eigen::VectorX<double>::Zero(6);
    out(0) = 0.25 * std::sin(context.get_time()) + 1;
    output_value = out;
  }
};

ConveyorSystem setupLCSPlant(const std::string& name, bool build = true) {
  drake::multibody::MultibodyPlantConfig config;
  config.time_step = 0.005;  // continuous plant
  config.penetration_allowance = 0.001;
  config.contact_model = "point";
  config.contact_surface_representation = "polygon";

  drake::geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  auto lcs_builder = std::make_unique<drake::systems::DiagramBuilder<double>>();
  auto [plant_lcs, scene_graph_lcs] = drake::multibody::AddMultibodyPlant(
      config, scene_graph_config, lcs_builder.get());
  std::string conveyor_belt_tool_url =
      "examples/conveyor_belt/conveyor_tool.urdf";
  std::string box_url = "examples/resources/conveyor_belt/box.sdf";
  drake::multibody::Parser parser(lcs_builder.get());
  parser.AddModels(conveyor_belt_tool_url);
  parser.AddModels(box_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamic modify them.
  const drake::multibody::RigidBody<double>& conveyor_belt_body =
      plant_lcs.GetBodyByName("conveyor_belt_tool");
  const drake::geometry::GeometryId geom_id =
      plant_lcs.GetCollisionGeometriesForBody(conveyor_belt_body).at(0);
  plant_lcs.DeclareSurfaceVelocityInputPort(
      geom_id, Eigen::Vector3d(0.0, 1.0, 0.0), 0.5);
  plant_lcs.set_name(name);
  plant_lcs.Finalize();

  std::unique_ptr<drake::systems::Diagram<double>> plant_diagram;
  if (build) plant_diagram = lcs_builder->Build();

  return {.builder = std::move(lcs_builder),
          .diagram = build ? std::move(plant_diagram) : nullptr,
          .plant = &plant_lcs,
          .scene_graph = &scene_graph_lcs};
}

std::vector<drake::SortedPair<drake::geometry::GeometryId>> extractContactPairs(
    const drake::multibody::MultibodyPlant<double>* plant) {
  std::vector<drake::SortedPair<drake::geometry::GeometryId>> contact_pairs;
  const drake::geometry::GeometryId geom_a =
      plant
          ->GetCollisionGeometriesForBody(
              plant->GetBodyByName("conveyor_belt_tool"))
          .at(0);
  const drake::geometry::GeometryId geom_b =
      plant->GetCollisionGeometriesForBody(plant->GetBodyByName("box")).at(0);
  const drake::geometry::GeometryId geom_c =
      plant->GetCollisionGeometriesForBody(plant->GetBodyByName("floor")).at(0);
  contact_pairs.push_back({geom_a, geom_b});
  contact_pairs.push_back({geom_b, geom_c});
  return contact_pairs;
}

int conveyor_belt_tool() {
  ConveyorSystem conveyor_lcs = setupLCSPlant("plant_for_lcs");
  ConveyorSystem conveyor_sim = setupLCSPlant("plant_for_sim", false);

  const auto prnt = [](const auto& e) { std::cout << e << std::endl; };
  auto u_ns = conveyor_lcs.plant->GetActuatorNames();
  std::cout << "inputs" << std::endl;
  std::for_each(u_ns.begin(), u_ns.end(), prnt);
  std::cout << "states" << std::endl;
  auto s_ns = conveyor_lcs.plant->GetStateNames();
  std::for_each(s_ns.begin(), s_ns.end(), prnt);

  // Get contact geometry pairs
  auto contact_pairs = extractContactPairs(conveyor_lcs.plant);

  // Create contexts for the plant and LCS factory system.
  std::unique_ptr<drake::systems::Context<double>> plant_diagram_context =
      conveyor_lcs.diagram->CreateDefaultContext();
  auto plant_autodiff =
      drake::systems::System<double>::ToAutoDiffXd(*conveyor_lcs.plant);
  auto& plant_for_lcs_context =
      conveyor_lcs.diagram->GetMutableSubsystemContext(
          *conveyor_lcs.plant, plant_diagram_context.get());
  auto plant_context_autodiff = plant_autodiff->CreateDefaultContext();

  // Add the LCS factory system.
  c3::systems::C3ControllerOptions options = drake::yaml::LoadYamlFile<
      c3::systems::C3ControllerOptions>(
      "examples/resources/conveyor_belt/conveyor_belt_tool_c3_options.yaml");
  auto lcs_factory_system =
      conveyor_sim.builder->AddSystem<c3::systems::LCSFactorySystem>(
          *conveyor_lcs.plant, plant_for_lcs_context, *plant_autodiff,
          *plant_context_autodiff, contact_pairs, options.lcs_factory_options);

  // Add the C3 controller.
  c3::C3::CostMatrices cost = c3::C3::CreateCostMatricesFromC3Options(
      options.c3_options, options.lcs_factory_options.N);
  auto c3_controller =
      conveyor_sim.builder->AddSystem<c3::systems::C3Controller>(
          *conveyor_lcs.plant, cost, options,
          lcs_factory_system->GetNumContactVelocityBiases());
  c3_controller->set_name("c3_controller");

  // Add a constant vector source for the desired state.
  Eigen::VectorXd xd(18);
  xd << 1, 1, 0.2, 1.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  auto xdes = conveyor_sim.builder
                  ->AddSystem<drake::systems::ConstantVectorSource<double>>(xd);

  // Add a vector-to-timestamped-vector converter.
  auto vector_to_timestamped_vector =
      conveyor_sim.builder->AddSystem<Vector2TimestampedVector>(18);

  // sim plant -> timestamped vector -> c3 controller
  conveyor_sim.builder->Connect(
      conveyor_sim.plant->get_state_output_port(),
      vector_to_timestamped_vector->get_input_port_state());
  conveyor_sim.builder->Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      c3_controller->get_input_port_lcs_state());

  // lcs factory system -> c3 controller's LCS input
  // x_des -> c3 controller's x_des input
  conveyor_sim.builder->Connect(lcs_factory_system->get_output_port_lcs(),
                                c3_controller->get_input_port_lcs());
  conveyor_sim.builder->Connect(xdes->get_output_port(),
                                c3_controller->get_input_port_target());

  // c3 controller's output -> plant inputs (actuators and surface velocity)
  auto c3_input = conveyor_sim.builder->AddSystem<C3Solution2Input>(6);
  conveyor_sim.builder->Connect(c3_controller->get_output_port_c3_solution(),
                                c3_input->get_input_port_c3_solution());
  const std::vector<int> state_demux_sizes = {5, 1};
  auto input_demux =
      conveyor_sim.builder->AddSystem<drake::systems::Demultiplexer>(
          state_demux_sizes);

  auto sine_vector_gen = conveyor_sim.builder->AddSystem<SineVectorGenerator>();
  conveyor_sim.builder->Connect(sine_vector_gen->get_output_port(),
                                input_demux->get_input_port());
  conveyor_sim.builder->Connect(input_demux->get_output_port(0),
                                conveyor_sim.plant->get_actuation_input_port());

  const drake::geometry::GeometryId geom_id =
      conveyor_sim.plant
          ->GetCollisionGeometriesForBody(
              conveyor_sim.plant->GetBodyByName("conveyor_belt_tool"))
          .at(0);
  conveyor_sim.builder->Connect(
      input_demux->get_output_port(1),
      conveyor_sim.plant->get_surface_speed_input_port(geom_id).value().get());

  // Add a ZeroOrderHold system for state updates.
  auto input_zero_order_hold =
      conveyor_sim.builder->AddSystem<drake::systems::ZeroOrderHold<double>>(
          1 / options.publish_frequency, 6);
  conveyor_sim.builder->Connect(c3_input->get_output_port_c3_input(),
                                input_zero_order_hold->get_input_port());
  conveyor_sim.builder->Connect(
      vector_to_timestamped_vector->get_output_port_timestamped_state(),
      lcs_factory_system->get_input_port_lcs_state());
  conveyor_sim.builder->Connect(input_zero_order_hold->get_output_port(),
                                lcs_factory_system->get_input_port_lcs_input());

  // Set up visualization
  auto meshcat = std::make_shared<drake::geometry::Meshcat>();
  drake::geometry::MeshcatVisualizer<double>::AddToBuilder(
      conveyor_sim.builder.get(), *conveyor_sim.scene_graph, meshcat);
  drake::geometry::MeshcatVisualizerParams meshcat_params;
  meshcat_params.delete_on_initialization_event = false;
  auto& visualizer = drake::geometry::MeshcatVisualizerd::AddToBuilder(
      conveyor_sim.builder.get(), *conveyor_sim.scene_graph, meshcat,
      std::move(meshcat_params));
  drake::multibody::meshcat::ContactVisualizerParams cparams;
  cparams.newtons_per_meter = 60.0;
  drake::multibody::meshcat::ContactVisualizerd::AddToBuilder(
      conveyor_sim.builder.get(), *conveyor_sim.plant, meshcat,
      std::move(cparams));

  // Setup state, input desired state loggers
  auto u_logger = drake::systems::LogVectorOutput(
      input_demux->get_output_port(0), conveyor_sim.builder.get());
  u_logger->set_name("u_logger");
  auto sim_state_logger = drake::systems::LogVectorOutput(
      conveyor_sim.plant->get_state_output_port(), conveyor_sim.builder.get());
  sim_state_logger->set_name("sim_state_logger");
  auto des_state_logger = drake::systems::LogVectorOutput(
       conveyor_sim.plant->get_net_actuation_output_port(), conveyor_sim.builder.get());
  des_state_logger->set_name("des_state_logger");

  // Set up context
  std::unique_ptr<drake::systems::Diagram<double>> diagram =
      conveyor_sim.builder->Build();
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());

  auto& plant_context = diagram->GetMutableSubsystemContext(
      *conveyor_sim.plant, diagram_context.get());

  // Force visualization
  diagram->ForcedPublish(*diagram_context);

  const std::string path =
      "/home/juanmedrano_eng/repos/c3/examples/conveyor_belt_tool_diagram.dot";
  std::ofstream graphviz(path);
  std::map<std::string, std::string> options_gv{{"plant/split", "I/O"}};
  graphviz << diagram->GetGraphvizString({}, options_gv);

  // Set up simulator
  drake::systems::Simulator<double> simulator(*diagram,
                                              std::move(diagram_context));
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  visualizer.StartRecording();
  simulator.AdvanceTo(5.0);
  visualizer.PublishRecording();

  // Plot data
  const auto& u_log = u_logger->FindLog(simulator.get_context());
  drake::common::CallPython("figure", 1);
  drake::common::CallPython("clf");
  drake::common::CallPython("plot", u_log.sample_times(),
                            u_log.data().transpose());
  // drake::common::CallPython("legend", drake::common::ToPythonTuple(
  //                                         "u_z", "u_x", "u_r", "u_y", "u_y", "s"));
  drake::common::CallPython("title", "Control input");

  const auto& state_log = sim_state_logger->FindLog(simulator.get_context());
  drake::common::CallPython("figure", 2);
  drake::common::CallPython("clf");
  drake::common::CallPython("plot", state_log.sample_times(),
                            state_log.data().transpose());
  // drake::common::CallPython("legend", drake::common::ToPythonTuple(
  //                                         "x", "z", "pitch", "vx", "vz", "wy"));
  drake::common::CallPython("title", "Sim Plant State");
  drake::common::CallPython("grid", true);

  const auto& des_state_log =
      des_state_logger->FindLog(simulator.get_context());
  drake::common::CallPython("figure", 3);
  drake::common::CallPython("clf");
  drake::common::CallPython("plot", des_state_log.sample_times(),
                            des_state_log.data().transpose());
  // drake::common::CallPython(
  //     "legend", drake::common::ToPythonTuple("x_d", "z_d", "pitch_d", "vx_d",
  //                                            "vz_d", "wy_d"));
  drake::common::CallPython("title", "Sim Plant Desired State");
  drake::common::CallPython("grid", true);

  return 0;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  conveyor_belt_tool();
}
