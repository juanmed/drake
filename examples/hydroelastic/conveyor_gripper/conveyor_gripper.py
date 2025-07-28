"""
This example demonstrates a gripper with a conveyor belt moving
objects around using contact.
"""

from pydrake.multibody.plant import MultibodyPlantConfig, AddMultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator, SimulatorConfig, ApplySimulatorConfig
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Meshcat
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)


def load_conveyor_gripper_scene():
    # Create a Meshcat instance for visualization.
    meshcat = Meshcat()

    # Create a DiagramBuilder.
    builder = DiagramBuilder()

    # Add MultibodyPlant and SceneGraph to the builder.
    plant_config = MultibodyPlantConfig(
        time_step=0.0,
        contact_model="hydroelastic",
        contact_surface_representation="polygon",
    )
    plant, scene_graph = AddMultibodyPlant(plant_config, builder)
    parser = Parser(plant)
    parser.AddModelsFromUrl(
        "package://drake/examples/hydroelastic/conveyor_gripper/conveyor_gripper_scene.sdf"
    )
    plant.Finalize()

    # Add MeshcatVisualizer for visualization.
    visualization_config = VisualizationConfig()
    visualization_config.publish_contacts = True
    visualization_config.publish_proximity = False
    visualization_config.enable_alpha_sliders = True
    visualization_config.publish_period = 0.05
    ApplyVisualizationConfig(visualization_config, builder, meshcat=meshcat)

    diagram = builder.Build()

    # Create a simulator.
    simulator_config = SimulatorConfig(
        target_realtime_rate=1.0, publish_every_time_step=True
    )
    simulator = Simulator(diagram)
    ApplySimulatorConfig(simulator_config, simulator)

    simulator.Initialize()
    meshcat.StartRecording()
    simulator.AdvanceTo(20.0)
    meshcat.PublishRecording()


if __name__ == "__main__":
    load_conveyor_gripper_scene()
