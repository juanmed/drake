#include <fstream>

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

int do_main_continous_plant() {
  multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  config.time_step = FLAGS_time_step;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;

  geometry::SceneGraphConfig scene_graph_config;
  scene_graph_config.default_proximity_properties.margin = 1e-3;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlant(config, scene_graph_config, &builder);
  std::string conveyor_belt_url =
      "package://drake/examples/conveyor_belt/conveyor_belt.sdf";
  multibody::Parser(&builder).AddModelsFromUrl(conveyor_belt_url);

  // Overrides the surface speed and surface velocity normal defined through
  // the sdf file, and also create their input ports to dynamic modify them.
  const multibody::RigidBody<double>& body =
      plant.GetBodyByName("conveyor_belt");
  const geometry::GeometryId geom_id =
      plant.GetCollisionGeometriesForBody(body).at(0);
  plant.DeclareSurfaceVelocityInputPort(geom_id, Vector3<double>(0.0, 1.0, 0.0),
                                        1.0);
  plant.Finalize();

  // Add a sine wave generator. This will be connected to the surface speed
  // input port.
  double amplitude = 1.0;
  double frequency = 1.0;
  double phase = 0.0;
  drake::systems::Sine<double>* sine_generator =
      builder.AddSystem<systems::Sine<double>>(amplitude, frequency, phase, 1);
  
  builder.Connect(sine_generator->get_output_port(0),
                  plant.get_surface_speed_input_port().value().get());
  
  // Uncomment the lines below to dynamically change the surface velocity normal                
  // auto sine_vector_gen = builder.AddSystem<SineVectorGenerator>();
  // builder.Connect(sine_vector_gen->get_output_port(),
  //                 plant.get_surface_velocity_normal_input_port().value().get());

  // Set up visualization
  auto meshcat = std::make_shared<geometry::Meshcat>();
  geometry::MeshcatVisualizer<double>::AddToBuilder(&builder, scene_graph,
                                                    meshcat);
  geometry::MeshcatVisualizerParams meshcat_params;
  meshcat_params.delete_on_initialization_event = false;
  auto& visualizer = geometry::MeshcatVisualizerd::AddToBuilder(
      &builder, scene_graph, meshcat, std::move(meshcat_params));
  multibody::meshcat::ContactVisualizerParams cparams;
  cparams.newtons_per_meter = 60.0;
  multibody::meshcat::ContactVisualizerd::AddToBuilder(&builder, plant, meshcat,
                                                       std::move(cparams));

  // Set up context
  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<systems::Context<double>> context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(context.get());

  // Force visualization
  diagram->ForcedPublish(*context);

  // Draw diagram of plant
  std::ofstream graphviz(FLAGS_graphviz);
  std::map<std::string, std::string> options{{"plant/split", "I/O"}};
  graphviz << diagram->GetGraphvizString({}, options);

  // Set up simulator
  systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  visualizer.StartRecording();
  simulator.AdvanceTo(20.0);
  visualizer.PublishRecording();

  return 0;
}

}  // namespace
}  // namespace conveyor_belt
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  // Initialize gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::conveyor_belt::do_main_continous_plant();
  return 0;
}