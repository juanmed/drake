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

namespace drake {
namespace examples {
namespace conveyor_belt {
namespace {

DEFINE_double(time_step, 0.0, "Simulation time step used for integrator.");
DEFINE_string(contact_model, "hydroelastic",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'. Default is 'polygon'.");
DEFINE_double(hydroelastic_modulus, 3.0e4,
              "Hydroelastic modulus of the ball, [Pa].");
DEFINE_double(resolution_hint_factor, 0.3,
              "This scaling factor, [unitless], multiplied by the radius of "
              "the ball gives the target edge length of the mesh of the ball "
              "on the surface of its hydroelastic representation. The smaller "
              "number gives a finer mesh with more tetrahedral elements.");
DEFINE_double(dissipation, 3.0,
              "Hunt & Crossley dissipation, [s/m], for the ball");
DEFINE_double(friction_coefficient, 0.3,
              "coefficient for both static and dynamic friction, [unitless], "
              "of the ball.");
DEFINE_string(
    graphviz, "/home/juaneng/repos/drake/conveyor_belt.dot",
    "Dump the Simulator's Diagram to this file in Graphviz format as a "
    "debugging aid");

int do_main_continous_plant() {
  multibody::MultibodyPlantConfig config;
  // We allow only discrete systems.
  config.time_step = FLAGS_time_step;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = multibody::AddMultibodyPlant(config, &builder);
  std::string conveyor_belt_url =
      "package://drake/examples/conveyor_belt/conveyor_belt.sdf";
  multibody::Parser(&builder).AddModelsFromUrl(conveyor_belt_url);

  const multibody::RigidBody<double>& body =
      plant.GetBodyByName("conveyor_belt");
  const geometry::GeometryId geom_id =
      plant.GetCollisionGeometriesForBody(body).at(0);
  plant.DeclareSurfaceVelocityInputPort(geom_id, Vector3<double>(0.0, 0.0, 0.0),
                                        1.0);
  plant.Finalize();

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