/// @file
///
/// Implements a simulation of the KUKA iiwa arm.  Like the driver for the
/// physical arm, this simulation communicates over LCM using lcmt_iiwa_status
/// and lcmt_iiwa_command messages. It is intended to be a be a direct
/// replacement for the KUKA iiwa driver and the actual robot hardware.

#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_common.h"
#include "drake/examples/kuka_iiwa_arm/iiwa_lcm.h"
#include "drake/examples/kuka_iiwa_arm/kuka_torque_controller.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcmt_iiwa_command.hpp"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/multibody/parsing/parser.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/controllers/state_feedback_controller_interface.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/systems/primitives/demultiplexer.h"
#include "drake/systems/primitives/discrete_derivative.h"
#include "drake/visualization/visualization_config_functions.h"

DEFINE_double(simulation_sec, std::numeric_limits<double>::infinity(),
              "Number of seconds to simulate.");
DEFINE_string(urdf, "", "Name of urdf to load");
DEFINE_double(target_realtime_rate, 1.0,
              "Playback speed.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_bool(torque_control, false, "Simulate using torque control mode.");
DEFINE_double(sim_dt, 3e-3,
              "The time step to use for MultibodyPlant model "
              "discretization.");

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace {
using multibody::MultibodyPlant;
using multibody::PackageMap;
using systems::Context;
using systems::Demultiplexer;
using systems::Simulator;
using systems::StateInterpolatorWithDiscreteDerivative;
using systems::controllers::InverseDynamicsController;
using systems::controllers::StateFeedbackControllerInterface;

int DoMain() {
  systems::DiagramBuilder<double> sim_builder;

  // Adds a sim_plant.
  auto [sim_plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&sim_builder, FLAGS_sim_dt);
  const char* kModelUrl =
      "package://drake_models/iiwa_description/"
      "urdf/iiwa14_polytope_collision.urdf";
  const std::string urdf =
      (!FLAGS_urdf.empty() ? FLAGS_urdf : PackageMap{}.ResolveUrl(kModelUrl));
  const multibody::ModelInstanceIndex iiwa_instance_sim =
      multibody::Parser(&sim_builder).AddModels(urdf).at(0);
  sim_plant.WeldFrames(sim_plant.world_frame(),
                       sim_plant.GetFrameByName("base"));
  const char* gripper_url =
      "package://drake_models/iiwa_description/"
      "urdf/iiwa14_polytope_collision.urdf";
  const std::string gripper_sdf = PackageMap{}.ResolveUrl(gripper_url);
  const multibody::ModelInstanceIndex gripper_instance =
      multibody::Parser(&sim_builder).AddModels(gripper_sdf).at(0);
  (void)gripper_instance;
  math::RigidTransformd X_WB(math::RollPitchYawd(0.0, 1.57079, 0.0),
                             Eigen::Vector3d(0.0, 0., 0.2));
  sim_plant.WeldFrames(sim_plant.GetFrameByName("iiwa_link_ee_kuka"),
                       sim_plant.GetFrameByName("gripper"), X_WB);

  auto iiwa_model_instance_sim = sim_plant.GetModelInstanceByName("iiwa14");
  sim_plant.RenameModelInstance(iiwa_model_instance_sim, "iiwa14_sim");
  sim_plant.Finalize();

  // Add a control plant
  systems::DiagramBuilder<double> builder;
  auto control_plant =
      builder.AddSystem<multibody::MultibodyPlant<double>>(0.0);
  multibody::Parser control_parser(&builder);
  auto iiwa_instance = control_parser.AddModels(urdf).at(0);
  control_plant->RenameModelInstance(iiwa_instance, "iiwa_robot_2");
  control_plant->Finalize();

  // TODO(sammy-tri) Add a floor.

  // Creates and adds LCM publisher for visualization.
  auto lcm = builder.AddSystem<systems::lcm::LcmInterfaceSystem>();
  visualization::ApplyVisualizationConfig({}, &builder, nullptr, nullptr,
                                          nullptr, nullptr, lcm);

  // Since we welded the model to the world above, the only remaining joints
  // should be those in the arm.
  const int num_joints = control_plant->num_positions();
  DRAKE_DEMAND(num_joints % kIiwaArmNumJoints == 0);
  const int num_iiwa = num_joints / kIiwaArmNumJoints;

  // Adds a iiwa controller.
  StateFeedbackControllerInterface<double>* controller = nullptr;
  if (FLAGS_torque_control) {
    VectorX<double> stiffness, damping_ratio;
    SetTorqueControlledIiwaGains(&stiffness, &damping_ratio);
    stiffness = stiffness.replicate(num_iiwa, 1).eval();
    damping_ratio = damping_ratio.replicate(num_iiwa, 1).eval();
    controller = builder.AddSystem<KukaTorqueController<double>>(
        *control_plant, stiffness, damping_ratio);
  } else {
    VectorX<double> iiwa_kp, iiwa_kd, iiwa_ki;
    SetPositionControlledIiwaGains(&iiwa_kp, &iiwa_ki, &iiwa_kd);
    iiwa_kp = iiwa_kp.replicate(num_iiwa, 1).eval();
    iiwa_kd = iiwa_kd.replicate(num_iiwa, 1).eval();
    iiwa_ki = iiwa_ki.replicate(num_iiwa, 1).eval();
    controller = builder.AddSystem<InverseDynamicsController<double>>(
        *control_plant, iiwa_kp, iiwa_ki, iiwa_kd,
        false /* without feedforward acceleration */);
  }

  // Create the command subscriber and status publisher.
  auto command_sub = builder.AddSystem(
      systems::lcm::LcmSubscriberSystem::Make<drake::lcmt_iiwa_command>(
          "IIWA_COMMAND", lcm));
  command_sub->set_name("command_subscriber");
  auto command_receiver = builder.AddSystem<IiwaCommandReceiver>(num_joints);
  command_receiver->set_name("command_receiver");
  auto plant_state_demux =
      builder.AddSystem<Demultiplexer>(2 * num_joints, num_joints);
  plant_state_demux->set_name("plant_state_demux");
  auto desired_state_from_position =
      builder.AddSystem<StateInterpolatorWithDiscreteDerivative>(
          num_joints, kIiwaLcmStatusPeriod,
          true /* suppress_initial_transient */);
  desired_state_from_position->set_name("desired_state_from_position");
  auto status_pub = builder.AddSystem(
      systems::lcm::LcmPublisherSystem::Make<lcmt_iiwa_status>(
          "IIWA_STATUS", lcm, kIiwaLcmStatusPeriod /* publish period */));
  status_pub->set_name("status_publisher");
  auto status_sender = builder.AddSystem<IiwaStatusSender>(num_joints);
  status_sender->set_name("status_sender");

  builder.Connect(command_sub->get_output_port(),
                  command_receiver->get_message_input_port());
  builder.Connect(plant_state_demux->get_output_port(0),
                  command_receiver->get_position_measured_input_port());
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  desired_state_from_position->get_input_port());
  builder.Connect(desired_state_from_position->get_output_port(),
                  controller->get_input_port_desired_state());
  builder.Connect(control_plant->get_state_output_port(iiwa_instance),
                  plant_state_demux->get_input_port(0));
  builder.Connect(plant_state_demux->get_output_port(0),
                  status_sender->get_position_measured_input_port());
  builder.Connect(plant_state_demux->get_output_port(1),
                  status_sender->get_velocity_estimated_input_port());
  builder.Connect(command_receiver->get_commanded_position_output_port(),
                  status_sender->get_position_commanded_input_port());
  builder.Connect(control_plant->get_state_output_port(),
                  controller->get_input_port_estimated_state());
  builder.Connect(controller->get_output_port_control(),
                  control_plant->get_actuation_input_port(iiwa_instance));
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_commanded_input_port());
  builder.Connect(controller->get_output_port_control(),
                  status_sender->get_torque_measured_input_port());
  // TODO(sammy-tri) Add a low-pass filter for simulated external torques.
  // This would slow the simulation significantly, however.  (see #12631)
  builder.Connect(
      control_plant->get_generalized_contact_forces_output_port(iiwa_instance),
      status_sender->get_torque_external_input_port());
  builder.Connect(status_sender->get_output_port(),
                  status_pub->get_input_port());
  // Connect the torque input in torque control
  if (FLAGS_torque_control) {
    KukaTorqueController<double>* torque_controller =
        dynamic_cast<KukaTorqueController<double>*>(controller);
    DRAKE_DEMAND(torque_controller != nullptr);
    builder.Connect(command_receiver->get_commanded_torque_output_port(),
                    torque_controller->get_input_port_commanded_torque());
  }

  // Connect control: robot state from full plant to controller plant
  builder.Connect(sim_plant.get_state_output_port(iiwa_instance_sim),
                  controller->get_input_port_estimated_state());
  builder.Connect(controller->get_output_port_control(),
                  sim_plant.get_actuation_input_port(iiwa_instance_sim));

  std::unique_ptr<systems::Diagram<double>> diagram = sim_builder.Build();

  builder.BuildInto(&(*diagram));

  auto sys = builder.Build();

  Simulator<double> simulator(*sys);

  simulator.set_publish_every_time_step(false);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();

  // Simulate for a very long time.
  simulator.AdvanceTo(FLAGS_simulation_sec);

  return 0;
}

}  // namespace
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::kuka_iiwa_arm::DoMain();
}
