#include <iostream>

#include <gtest/gtest.h>

#include "drake/geometry/geometry_ids.h"
#include "drake/geometry/proximity_properties.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace multibody {
namespace {

GTEST_TEST(SurfaceVelocityTest, BoxSurfaceVelocity) {
  constexpr double tol = 1e-5;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);

  CoulombFriction<double> friction(0.5, 0.3);

  // Create ground
  geometry::GeometryId ground_id = plant.RegisterCollisionGeometry(
      plant.world_body(),
      geometry::HalfSpace::MakePose(Eigen::Vector3d::UnitY(),
                                    Eigen::Vector3d::Zero()),
      geometry::HalfSpace(), "floor", friction);

  // Crate an thin, elongated box that ressembles a conveyor belt
  const RigidBody<double>& belt =
      plant.AddRigidBody("belt", SpatialInertia<double>::MakeUnitary());
  const double belt_stiffness = 980;
  const double belt_dissipation = 3.2;
  const double surface_speed = 0.5;
  const Eigen::Vector3d velocity_n(1.0, 0.0, 0.0);

  geometry::ProximityProperties belt_props;
  belt_props.AddProperty(geometry::internal::kMaterialGroup,
                         geometry::internal::kFriction, friction);
  belt_props.AddProperty(geometry::internal::kMaterialGroup,
                         geometry::internal::kPointStiffness, belt_stiffness);
  belt_props.AddProperty(geometry::internal::kMaterialGroup,
                         geometry::internal::kHcDissipation, belt_dissipation);
  belt_props.AddProperty(geometry::internal::kSurfaceVelocityGroup,
                         geometry::internal::kSurfaceSpeed, surface_speed);
  belt_props.AddProperty(geometry::internal::kSurfaceVelocityGroup,
                         geometry::internal::kSurfaceVelocityNormal,
                         velocity_n);
  const double w = 3;
  const double d = 0.5;
  const double h = 0.1;
  geometry::GeometryId belt_geom_id = plant.RegisterCollisionGeometry(
      belt, math::RigidTransformd(Eigen::Vector3d(0., 0., 0.)),
      geometry::Box(Eigen::Vector3d(w, d, h)), "belt_collision",
      std::move(belt_props));

  plant.Finalize();
  std::unique_ptr<drake::systems::Context<double>> context =
      plant.CreateDefaultContext();
  auto diagram = builder.Build();

  // Set pose of body (and read it again just to make sure this
  // is doing what it is supposed to do).
  const double yaw = 0.78;
  math::RigidTransformd body_pose(math::RollPitchYaw(0., 0., yaw),
                                  Eigen::Vector3d(0., 0., 1.));
  plant.SetFreeBodyPoseInWorldFrame(&(*context), belt, body_pose);
  math::RigidTransformd pose = plant.GetFreeBodyPose(*context, belt);

  // Assume there are some contacts on each face of the conveyor belt.
  // For ease of reading, these are expressed in coordinates of its body
  // frame and later will be transformed to world coordinates
  std::vector<Eigen::Vector3d> contacts_G = {
      {w / 2, 0., 0.},    // Contact at +x face
      {-w / 2, 0., 0.},   // Contact at -x face
      {0., d / 2, 0.},    // Contact at +y face
      {0., -d / 2, 0.},   // Contact at -y face
      {0., 0., h / 2},    // Contact at +z face
      {0., 0., -h / 2}};  // Contact at -z face

  for (const Eigen::Vector3d& c_G : contacts_G) {
    const Eigen::Vector3d c_W = pose * c_G;
    Eigen::Vector3d surface_v = plant.GetSurfaceVelocity(
        *context, belt_geom_id, scene_graph.model_inspector(), pose, c_W);

    // Verify the direction of surface velocity is equal to cross product
    // between the surface normal at each contact point and the velocity
    // normal vector
    Eigen::Vector3d v_ref_W =
        surface_speed *
        (body_pose.rotation() * velocity_n.cross(c_G.normalized()));
    Eigen::Vector3d v_ss_W = pose.rotation() * surface_v;
    EXPECT_LT((v_ss_W - v_ref_W).norm(), tol);

    // When the velocity normal and the surface normal vectors are parallel,
    // their cross product is 0 meaning the surface velocity should be also
    // very close to 0. This occurs in this case for a contact at the +x and -x
    // faces because they are parallel to the velocity normal.
    EXPECT_NEAR(v_ss_W.norm(), v_ref_W.norm() > 1e-3 ? surface_speed : 0., tol);
  }

  (void)ground_id;
  (void)belt_geom_id;
}

GTEST_TEST(SurfaceVelocityTest, BoxSurfaceVelocityFromSDF) {
  const double tol = 1e-5;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
  std::string conveyor_belt_url =
      "package://drake/examples/conveyor_belt/conveyor_belt_simple.sdf";
  multibody::Parser(&builder).AddModelsFromUrl(conveyor_belt_url);
  plant.Finalize();

  std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  auto& context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Set conveyor belt's pose
  const multibody::RigidBody<double>& belt =
      plant.GetBodyByName("conveyor_belt");
  math::RigidTransformd body_pose = plant.EvalBodyPoseInWorld(context, belt);
  const geometry::GeometryId belt_geom_id =
      plant.GetCollisionGeometriesForBody(belt).at(0);

  // Assume there are some contacts on each face of the conveyor belt.
  // For ease of reading, these are expressed in coordinates of its body
  // frame and later will be transformed to world coordinates
  const double w = 10;
  const double d = 1.0;
  const double h = 0.1;
  std::vector<Eigen::Vector3d> contacts_G = {
      {w / 2, 0., 0.},    // Contact at +x face
      {-w / 2, 0., 0.},   // Contact at -x face
      {0., d / 2, 0.},    // Contact at +y face
      {0., -d / 2, 0.},   // Contact at -y face
      {0., 0., h / 2},    // Contact at +z face
      {0., 0., -h / 2}};  // Contact at -z face

  const geometry::SceneGraphInspector<double>& inspector =
      plant.EvalSceneGraphInspector(context);
  std::optional<Eigen::Vector3d> maybe_surface_speed =
      plant.GetSurfaceSpeedAndNormal(
          context, belt_geom_id, inspector,
          drake::math::RigidTransform<double>::Identity());

  // Verify surface speed and normal exists
  EXPECT_TRUE(maybe_surface_speed.has_value());
  const double surface_speed = maybe_surface_speed.value().norm();
  const Eigen::Vector3d velocity_n = maybe_surface_speed.value().normalized();

  for (const Eigen::Vector3d& c_G : contacts_G) {
    const Eigen::Vector3d c_W = body_pose * c_G;
    Eigen::Vector3d surface_v = plant.GetSurfaceVelocity(
        context, belt_geom_id, scene_graph.model_inspector(), body_pose, c_W);

    // Verify the direction of surface velocity is equal to cross product
    // between the surface normal at each contact point and the velocity
    // normal vector
    Eigen::Vector3d v_ref_W =
        surface_speed *
        (body_pose.rotation() * velocity_n.cross(c_G.normalized()));
    Eigen::Vector3d v_ss_W = body_pose.rotation() * surface_v;
    EXPECT_LT((v_ss_W - v_ref_W).norm(), tol);

    // When the velocity normal and the surface normal vectors are parallel,
    // their cross product is 0 meaning the surface velocity should be also
    // very close to 0. This occurs in this case for a contact at the +x and -x
    // faces because they are parallel to the velocity normal.
    EXPECT_NEAR(v_ss_W.norm(), v_ref_W.norm() > 1e-3 ? surface_speed : 0., tol);
  }
}

GTEST_TEST(SurfaceVelocityTest, BoxSurfaceVelocityFromSDFwithInput) {
  const double tol = 1e-5;

  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      multibody::AddMultibodyPlantSceneGraph(&builder, 0.0);
  std::string conveyor_belt_url =
      "package://drake/examples/conveyor_belt/conveyor_belt_simple.sdf";
  multibody::Parser(&builder).AddModelsFromUrl(conveyor_belt_url);

  // Set conveyor belt's pose
  const multibody::RigidBody<double>& belt =
      plant.GetBodyByName("conveyor_belt");
  const geometry::GeometryId belt_geom_id =
      plant.GetCollisionGeometriesForBody(belt).at(0);
  plant.DeclareSurfaceVelocityInputPort(belt_geom_id,
                                        Vector3<double>(0.0, 1.0, 0.0), 0.25);
  plant.Finalize();

  std::unique_ptr<drake::systems::Diagram<double>> diagram = builder.Build();
  std::unique_ptr<drake::systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  auto& context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  math::RigidTransformd body_pose = plant.EvalBodyPoseInWorld(context, belt);

  // Assume there are some contacts on each face of the conveyor belt.
  // For ease of reading, these are expressed in coordinates of its body
  // frame and later will be transformed to world coordinates
  const double w = 10;
  const double d = 1.0;
  const double h = 0.1;
  std::vector<Eigen::Vector3d> contacts_G = {
      {w / 2, 0., 0.},    // Contact at +x face
      {-w / 2, 0., 0.},   // Contact at -x face
      {0., d / 2, 0.},    // Contact at +y face
      {0., -d / 2, 0.},   // Contact at -y face
      {0., 0., h / 2},    // Contact at +z face
      {0., 0., -h / 2}};  // Contact at -z face

  const geometry::SceneGraphInspector<double>& inspector =
      plant.EvalSceneGraphInspector(context);
  std::optional<Eigen::Vector3d> maybe_surface_speed =
      plant.GetSurfaceSpeedAndNormal(
          context, belt_geom_id, inspector,
          drake::math::RigidTransform<double>::Identity());

  // Verify surface speed and normal exists
  EXPECT_TRUE(maybe_surface_speed.has_value());
  const double surface_speed = maybe_surface_speed.value().norm();
  const Eigen::Vector3d velocity_n = maybe_surface_speed.value().normalized();

  for (const Eigen::Vector3d& c_G : contacts_G) {
    const Eigen::Vector3d c_W = body_pose * c_G;
    Eigen::Vector3d surface_v = plant.GetSurfaceVelocity(
        context, belt_geom_id, scene_graph.model_inspector(), body_pose, c_W);

    // Verify the direction of surface velocity is equal to cross product
    // between the surface normal at each contact point and the velocity
    // normal vector
    Eigen::Vector3d v_ref_W =
        surface_speed *
        (body_pose.rotation() * velocity_n.cross(c_G.normalized()));
    Eigen::Vector3d v_ss_W = body_pose.rotation() * surface_v;
    EXPECT_LT((v_ss_W - v_ref_W).norm(), tol);

    // When the velocity normal and the surface normal vectors are parallel,
    // their cross product is 0 meaning the surface velocity should be also
    // very close to 0. This occurs in this case for a contact at the +x and -x
    // faces because they are parallel to the velocity normal.
    EXPECT_NEAR(v_ss_W.norm(), v_ref_W.norm() > 1e-3 ? surface_speed : 0., tol);
  }
}

}  // namespace
}  // namespace multibody
}  // namespace drake