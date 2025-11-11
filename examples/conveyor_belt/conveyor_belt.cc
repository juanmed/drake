#include <fstream>
#include <iostream>

#include <gflags/gflags.h>

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
#include "drake/systems/primitives/sine.h"
#include "drake/geometry/proximity_properties.h"


#include "examples/lcs/lcs_factory_options.h"
#include "examples/lcs/lcs_factory_system.h"
#include "examples/lcs/timestamped_vector.h"

using drake::geometry::GeometryId;
using drake::geometry::SceneGraph;
using drake::geometry::Sphere;
using drake::geometry::internal::kSurfaceSpeed;
using drake::geometry::internal::kSurfaceVelocityGroup;
using drake::geometry::internal::kSurfaceVelocityNormal;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::systems::Diagram;
using drake::systems::DiagramBuilder;
using drake::systems::System;

namespace drake {
namespace examples {
namespace conveyor_belt {
namespace {

DEFINE_double(time_step, 0.003, "Simulation time step used for integrator.");
DEFINE_string(contact_model, "hydroelastic",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'. Default is 'polygon'.");
DEFINE_string(
    graphviz, "/home/juaneng/repos/drake/conveyor_belt.dot",
    "Dump the Simulator's Diagram to this file in Graphviz format as a "
    "debugging aid");

class SineVectorGenerator : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SineVectorGenerator);
  SineVectorGenerator() {
    this->DeclareVectorOutputPort("sine_cosine",
                                  systems::BasicVector<double>(3),
                                  &SineVectorGenerator::calc_output);
  }

  void calc_output(const systems::Context<double>& context,
                   systems::BasicVector<double>* output_vector) const {
    Eigen::VectorBlock<Eigen::VectorX<double>> output_value =
        output_vector->get_mutable_value();
    Vector3<double> out = Vector3<double>::Zero();
    out.x() = std::sin(context.get_time());
    out.y() = std::cos(context.get_time());
    output_value = out;
  }
};

// int do_main_continous_plant() {
//   multibody::MultibodyPlantConfig config;
//   // We allow only discrete systems.
//   config.time_step = FLAGS_time_step;
//   config.penetration_allowance = 0.001;
//   config.contact_model = FLAGS_contact_model;
//   config.contact_surface_representation = FLAGS_contact_surface_representation;

//   geometry::SceneGraphConfig scene_graph_config;
//   scene_graph_config.default_proximity_properties.margin = 1e-3;

//   systems::DiagramBuilder<double> builder;
//   auto [plant, scene_graph] =
//       multibody::AddMultibodyPlant(config, scene_graph_config, &builder);
//   std::string conveyor_belt_url =
//       "package://drake/examples/conveyor_belt/conveyor_belt.sdf";
//   multibody::Parser(&builder).AddModelsFromUrl(conveyor_belt_url);

//   // Overrides the surface speed and surface velocity normal defined through
//   // the sdf file, and also create their input ports to dynamic modify them.
//   const multibody::RigidBody<double>& body =
//       plant.GetBodyByName("conveyor_belt");
//   const geometry::GeometryId geom_id =
//       plant.GetCollisionGeometriesForBody(body).at(0);
//   plant.DeclareSurfaceVelocityInputPort(geom_id, Vector3<double>(0.0, 1.0, 0.0),
//                                         1.0);
//   plant.Finalize();

//   // Add a sine wave generator. This will be connected to the surface speed
//   // input port.
//   double amplitude = 1.0;
//   double frequency = 1.0;
//   double phase = 0.0;
//   drake::systems::Sine<double>* sine_generator =
//       builder.AddSystem<systems::Sine<double>>(amplitude, frequency, phase, 1);
  
//   builder.Connect(sine_generator->get_output_port(0),
//                   plant.get_surface_speed_input_port(geom_id).value().get());
  
//   // Uncomment the lines below to dynamically change the surface velocity normal                
//   // auto sine_vector_gen = builder.AddSystem<SineVectorGenerator>();
//   // builder.Connect(sine_vector_gen->get_output_port(),
//   //                 plant.get_surface_velocity_normal_input_port().value().get());

//   // Set up visualization
//   auto meshcat = std::make_shared<geometry::Meshcat>();
//   geometry::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph,
//                                                     meshcat);
//   geometry::MeshcatVisualizerParams meshcat_params;
//   meshcat_params.delete_on_initialization_event = false;
//   auto& visualizer = geometry::MeshcatVisualizerd::AddToBuilder(
//       &builder, scene_graph, meshcat, std::move(meshcat_params));
//   multibody::meshcat::ContactVisualizerParams cparams;
//   cparams.newtons_per_meter = 60.0;
//   multibody::meshcat::ContactVisualizerd::AddToBuilder(&builder, plant, meshcat,
//                                                        std::move(cparams));

//   // Set up context
//   std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
//   std::unique_ptr<systems::Context<double>> context =
//       diagram->CreateDefaultContext();
//   diagram->SetDefaultContext(context.get());

//   // Force visualization
//   diagram->ForcedPublish(*context);

//   // Draw diagram of plant
//   std::ofstream graphviz(FLAGS_graphviz);
//   std::map<std::string, std::string> options{{"plant/split", "I/O"}};
//   graphviz << diagram->GetGraphvizString({}, options);

//   // Set up simulator
//   systems::Simulator<double> simulator(*diagram);
//   simulator.set_target_realtime_rate(1.0);
//   simulator.Initialize();
//   visualizer.StartRecording();
//   simulator.AdvanceTo(20.0);
//   visualizer.PublishRecording();

//   return 0;
// }

int do_main() {

  DiagramBuilder<double> builder_;
  MultibodyPlant<double>* plant_{nullptr};
  SceneGraph<double>* scene_graph_{nullptr};
  std::unique_ptr<Diagram<double>> diagram_;
  std::unique_ptr<Context<double>> diagram_context_;
  Context<double>* plant_context_{nullptr};
  std::unique_ptr<MultibodyPlant<drake::AutoDiffXd>> plant_autodiff_;
  std::unique_ptr<Context<drake::AutoDiffXd>> plant_autodiff_context_;
  c3::LCSFactoryOptions options_;
  std::vector<drake::SortedPair<drake::geometry::GeometryId>>
      contact_geometries_;
  // std::unique_ptr<LCSFactory> lcs_factory_;
  drake::geometry::GeometryId conveyor_belt_geometry_id_;
  drake::geometry::GeometryId sphere_geometry_id_;

  std::tie(plant_, scene_graph_) =
      AddMultibodyPlantSceneGraph(&builder_, 0.0);

  drake::multibody::Parser parser(plant_, scene_graph_);
  parser.AddModelsFromUrl("package://drake/examples/conveyor_belt/conveyor_belt_ex.sdf");
  plant_->Finalize();

  diagram_ = builder_.Build();
  diagram_context_ = diagram_->CreateDefaultContext();
  plant_context_ =
      &diagram_->GetMutableSubsystemContext(*plant_, diagram_context_.get());

  plant_autodiff_ = System<double>::ToAutoDiffXd(*plant_);
  plant_autodiff_context_ = plant_autodiff_->CreateDefaultContext();

  // Retrieve collision geometries for relevant bodies.
  std::vector<GeometryId> conveyor_belt_collision_geoms =
      plant_->GetCollisionGeometriesForBody(
          plant_->GetBodyByName("conveyor_belt"));
  std::vector<GeometryId> sphere_collision_geoms =
      plant_->GetCollisionGeometriesForBody(plant_->GetBodyByName("sphere"));

  conveyor_belt_geometry_id_ = conveyor_belt_collision_geoms[0];
  sphere_geometry_id_ = sphere_collision_geoms[0];
  contact_geometries_.emplace_back(conveyor_belt_geometry_id_,
                                    sphere_geometry_id_);

  options_.contact_model = "stewart_and_trinkle";
  options_.num_contacts = 1;
  options_.num_friction_directions =
      2;  // Square approximation to cone friction
  options_.spring_stiffness = 1.0;
  options_.mu = {0.5};
  options_.N = 1;
  options_.dt = 0.01;

  // Create some state and input vectors to update the LCS
  // Make sure to not zero all elements of state because some correspond
  // to orientation, which an throw if an ill-formed element is passed
  const auto q0 = plant_->GetPositions(*plant_context_);
  const auto v0 = plant_->GetVelocities(*plant_context_);
  drake::VectorX<double> state(q0.size() + v0.size());
  state << q0, v0;
  drake::VectorX<double> input = drake::VectorX<double>::Zero(plant_->num_actuators());

  // lcs_factory_ = std::make_unique<LCSFactory>(
  //     *plant_, *plant_context_, *plant_autodiff_, *plant_autodiff_context_,
  //     contact_geometries_, options_);
  // lcs_factory_->UpdateStateAndInput(state, input);

  std::unique_ptr<c3::systems::LCSFactorySystem> lcs_factory_system;
  std::unique_ptr<drake::systems::Context<double>> lcs_context;
  std::unique_ptr<drake::systems::SystemOutput<double>> lcs_output;

  // Construct LCSFactorySystem
  lcs_factory_system = std::make_unique<c3::systems::LCSFactorySystem>(
      *plant_, *plant_context_, *plant_autodiff_, *plant_autodiff_context_,
      contact_geometries_, options_);
  // int n_b = lcs_factory_->GetNumContactVelocityBiases(*plant_, *plant_context_,
  //                                                     contact_geometries_);
  lcs_context = lcs_factory_system->CreateDefaultContext();
  lcs_output = lcs_factory_system->AllocateOutput();

  // Set up dummy state and input
  auto state_vec = c3::systems::TimestampedVector<double>(
      plant_->num_positions() + plant_->num_velocities());
  const auto q0_t = plant_->GetPositions(*plant_context_);
  const auto v0_t = plant_->GetVelocities(*plant_context_);

  Eigen::VectorXd x(q0_t.size() + v0_t.size());
  x << q0_t, v0_t;
  state_vec.SetDataVector(x);
  state_vec.set_timestamp(0.0);
  lcs_factory_system->get_input_port_lcs_state().FixValue(lcs_context.get(),
                                                          state_vec);

  Eigen::VectorXd u = Eigen::VectorXd::Zero(plant_->num_actuators() + 1);
  lcs_factory_system->get_input_port_lcs_input().FixValue(lcs_context.get(),
                                                          u);
  // Should not throw and should produce an LCS object with correct dimensions
  if (lcs_output) {
    std::cout << "existe lcs output: " << lcs_output->num_ports() << std::endl;
  } else {
    std::cout << "no existe lcs context" << std::endl;
  }
  if(lcs_context) {
    std::cout << "existe lcs context" << std::endl;
  } else {
    std::cout << "no existe lcs context" << std::endl;
  }
  lcs_factory_system->CalcOutput(*lcs_context, lcs_output.get());


  return 0;
}


}  // namespace
}  // namespace conveyor_belt
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  // Initialize gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::conveyor_belt::do_main();
  return 0;
}