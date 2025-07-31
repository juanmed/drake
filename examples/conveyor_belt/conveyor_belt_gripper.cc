
#include <fstream>

#include <gflags/gflags.h>

#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/manipulation/kuka_iiwa/iiwa_command_receiver.h"
#include "drake/manipulation/kuka_iiwa/iiwa_constants.h"
#include "drake/manipulation/kuka_iiwa/iiwa_status_sender.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/plant/multibody_plant_config.h"
#include "drake/multibody/plant/multibody_plant_config_functions.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {
namespace examples {
namespace conveyor_belt_gripper {
namespace {

DEFINE_double(time_step, 1e-3, "Simulation time step");
DEFINE_string(
    graphviz, "/home/juaneng/repos/drake/conveyor_arm.dot",
    "Dump the Simulator's Diagram to this file in Graphviz format as a "
    "debugging aid");
DEFINE_string(contact_model, "hydroelastic_with_fallback",
              "Contact model. Options are: 'point', 'hydroelastic', "
              "'hydroelastic_with_fallback'.");
DEFINE_string(contact_surface_representation, "polygon",
              "Contact-surface representation for hydroelastics. "
              "Options are: 'triangle' or 'polygon'. Default is 'polygon'.");

/// Reorders the generalized force output vector of the ID controller
/// (internally using a control plant with only the gripper) to match the
/// actuation input ordering for the full simulation plant (containing gripper
/// and brick).
class GeneralizedForceToActuationOrdering : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GeneralizedForceToActuationOrdering);
  explicit GeneralizedForceToActuationOrdering(
      const multibody::MultibodyPlant<double>& plant)
      : Binv_(plant.MakeActuationMatrix().inverse()) {
    this->DeclareVectorInputPort("tau", plant.num_actuators());
    this->DeclareVectorOutputPort(
        "u", plant.num_actuators(),
        &GeneralizedForceToActuationOrdering::remap_output);
  }

  void remap_output(const systems::Context<double>& context,
                    systems::BasicVector<double>* output_vector) const {
    Eigen::VectorBlock<VectorX<double>> output_value =
        output_vector->get_mutable_value();
    const Eigen::VectorX<double>& input_value =
        this->EvalVectorInput(context, 0)->value();

    output_value.setZero();
    output_value = Binv_ * input_value;
  }

 private:
  const MatrixX<double> Binv_;
};

int DoMain() {
  multibody::MultibodyPlantConfig config;
  config.time_step = FLAGS_time_step;
  config.penetration_allowance = 0.001;
  config.contact_model = FLAGS_contact_model;
  config.contact_surface_representation = FLAGS_contact_surface_representation;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] = multibody::AddMultibodyPlant(config, &builder);

  // Location of models
  const std::string iiwa_url =
      "package://drake_models/iiwa_description/urdf/"
      "iiwa14_polytope_collision.urdf";
  const std::string belt_gripper_url =
      "package://drake/examples/hydroelastic/conveyor_gripper/"
      "conveyor_gripper_scene.sdf";

  multibody::Parser parser(&builder);
  // Load iiwa robot and fix its base to the world frame
  auto iiwa_instance = parser.AddModelsFromUrl(iiwa_url);
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"));
  // Load gripper and fix it to the robots end-effector frame
  auto gripper_instance = parser.AddModelsFromUrl(belt_gripper_url);
  math::RigidTransformd X_WB(math::RollPitchYawd(0.0, 1.57079, 0.0),
                             Eigen::Vector3d(0.0, 0., 0.2));
  plant.WeldFrames(plant.GetFrameByName("iiwa_link_ee_kuka"),
                   plant.GetFrameByName("gripper"), X_WB);
  plant.Finalize();

  // Create a model of the plant that will be controlled.
  multibody::MultibodyPlant<double> control_plant(FLAGS_time_step);
  multibody::Parser(&control_plant).AddModelsFromUrl(iiwa_url);
  control_plant.WeldFrames(control_plant.world_frame(),
                           control_plant.GetFrameByName("base"));
  control_plant.Finalize();

  // Add a controller that will use the control plant as a model of the robot.
  systems::controllers::StateFeedbackControllerInterface<double>* controller =
      nullptr;
  Eigen::VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
  kuka_iiwa_arm::SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
  const int num_joints = control_plant.num_positions();
  DRAKE_DEMAND(num_joints % manipulation::kuka_iiwa::kIiwaArmNumJoints == 0);
  const int num_iiwa = num_joints / manipulation::kuka_iiwa::kIiwaArmNumJoints;
  iiwa_kp = iiwa_kp.replicate(num_iiwa, 1).eval();
  iiwa_kd = iiwa_kd.replicate(num_iiwa, 1).eval();
  iiwa_ki = iiwa_ki.replicate(num_iiwa, 1).eval();
  controller =
      builder
          .AddSystem<systems::controllers::InverseDynamicsController<double>>(
              control_plant, iiwa_kp, iiwa_ki, iiwa_kd, false);

  // Connect the ID controller state estimate input to the state output of the
  // simulated iiwa arm.
  builder.Connect(plant.get_state_output_port(iiwa_instance.at(0)),
                  controller->get_input_port_estimated_state());

  // The inverse dynamics controller internally uses a "controlled plant",
  // which contains the gripper model *only* (i.e., no brick). Therefore, its
  // output must be re-mapped to the actuation input of the full "simulation
  // plant", which contains both gripper and brick. The system
  // GeneralizedForceToActuationOrdering fills this role.
  auto force_to_actuation =
      builder.AddSystem<GeneralizedForceToActuationOrdering>(control_plant);
  builder.Connect(controller->get_output_port_control(),
                  force_to_actuation->get_input_port());
  builder.Connect(force_to_actuation->get_output_port(0),
                  plant.get_actuation_input_port(iiwa_instance.at(0)));

  // Add LCM interface system.
  systems::lcm::LcmInterfaceSystem* lcm =
      builder.AddSystem<systems::lcm::LcmInterfaceSystem>();

  // Visualization config.
  visualization::VisualizationConfig vis_config;
  vis_config.publish_contacts = true;
  visualization::ApplyVisualizationConfig(vis_config, &builder, nullptr,
                                          nullptr, nullptr, nullptr, lcm);

  // Create a subscriber that receives the desired state from LCM network.
  auto command_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm));
  command_sub->set_name("command_subscriber");
  // Create a receiver for the commands received from the command subscriber
  auto command_receiver =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaCommandReceiver>(
          num_joints);
  command_receiver->set_name("command_receiver");
  // Create a demultiplexer for ?
  auto plant_state_demux =
      builder.AddSystem<systems::Demultiplexer>(2 * num_joints, num_joints);
  plant_state_demux->set_name("plant_state_demux");
  // This systems will compute the derivative of the commanded position which
  // is the commanded velocity.
  auto desired_state_from_position =
      builder.AddSystem<systems::StateInterpolatorWithDiscreteDerivative>(
          num_joints, manipulation::kuka_iiwa::kIiwaLcmStatusPeriod, true);
  desired_state_from_position->set_name("desired_state_from_position");

  // Connect the command receiver input to the command subscriber output.
  builder.Connect(command_sub->get_output_port(),
                  command_receiver->get_message_input_port());
  // Connect the command receiver's position_measured input port to the
  // plant demux's output port.
  builder.Connect(plant_state_demux->get_output_port(0),
                  command_receiver->get_position_measured_input_port());
  // Connect the received position command to the system that will compute
  // the velocity command by discretely deriving the position command.
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  desired_state_from_position->get_input_port());
  // Connect the desired state (position and velocity) to the input of the
  // controller
  builder.Connect(desired_state_from_position->get_output_port(),
                  controller->get_input_port_desired_state());
  // Connect the output state of the plant to the input of the demultiplexer.
  builder.Connect(plant.get_state_output_port(iiwa_instance.at(0)),
                  plant_state_demux->get_input_port(0));

  // Systems to pass along the status of the plant
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>(
          "IIWA_STATUS", lcm,
          manipulation::kuka_iiwa::kIiwaLcmStatusPeriod /* publish period
          */));
  status_pub->set_name("status_publisher");
  auto status_sender =
      builder.AddSystem<manipulation::kuka_iiwa::IiwaStatusSender>(num_joints);
  status_sender->set_name("status_sender");
  builder.Connect(plant_state_demux->get_output_port(0),
                  status_sender->get_position_measured_input_port());
  builder.Connect(plant_state_demux->get_output_port(1),
                  status_sender->get_velocity_estimated_input_port());
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  status_sender->get_position_commanded_input_port());
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_commanded_input_port());
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_measured_input_port());
  builder.Connect(
      plant.get_generalized_contact_forces_output_port(iiwa_instance.at(0)),
      status_sender->get_torque_external_input_port());
  builder.Connect(status_sender->get_output_port(),
                  status_pub->get_input_port());

  std::unique_ptr<systems::Diagram<double>> diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);
  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(std::numeric_limits<double>::infinity());

  // Draw diagram of plant
  std::ofstream graphviz(FLAGS_graphviz);
  std::map<std::string, std::string> options{{"plant/split", "I/O"}};
  graphviz << diagram->GetGraphvizString({}, options);

  // Exit without error
  return 0;
}

}  // namespace
}  // namespace conveyor_belt_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  // Initialize gflags.
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::examples::conveyor_belt_gripper::DoMain();
}
