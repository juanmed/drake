#include "lcs_factory.h"

#include <iostream>

#include "geom_geom_collider.h"
#include "multibody_utils.h"

#include "drake/common/text_logging.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/rotation_matrix.h"
#include "drake/solvers/moby_lcp_solver.h"

using std::set;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::SortedPair;
using drake::VectorX;
using drake::geometry::GeometryId;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::multibody::MultibodyForces;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;

using Eigen::MatrixXd;
using Eigen::Vector3;
using Eigen::VectorXd;

namespace c3 {
namespace multibody {

LCSFactory::LCSFactory(
    const MultibodyPlant<double>& plant, Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad, Context<AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSFactoryOptions& options)
    : plant_(plant),
      context_(context),
      plant_ad_(plant_ad),
      context_ad_(context_ad),
      contact_pairs_(contact_geoms),
      options_(options),
      n_q_(plant_.num_positions()),
      n_v_(plant_.num_velocities()),
      n_x_(n_q_ + n_v_),
      n_lambda_(multibody::LCSFactory::GetNumContactVariables(options)),
      n_u_(plant_.num_actuators()),
      n_contacts_(contact_geoms.size()),
      n_friction_directions_(options_.num_friction_directions),
      contact_model_(GetContactModelMap().at(options.contact_model)),
      mu_(options.mu),
      frictionless_(contact_model_ == ContactModel::kFrictionlessSpring),
      dt_(options_.dt),
      n_b_(multibody::LCSFactory::GetNumContactVelocityBiases(plant, context,
                                                              contact_geoms)),
      inspector_(
          multibody::LCSFactory::getSceneGraphInspector(plant, context)) {
  ComputeSetOfGeometriesWithSurfaceVelocity();
}

void LCSFactory::ComputeContactJacobian(VectorXd& phi, MatrixXd& Jn,
                                        MatrixXd& Jt) {
  phi.resize(n_contacts_);       // Signed distance values for contacts
  Jn.resize(n_contacts_, n_v_);  // Normal contact Jacobian
  Jt.resize(2 * n_contacts_ * n_friction_directions_,
            n_v_);  // Tangential contact Jacobian

  double phi_i;
  MatrixX<double> J_i;
  for (int i = 0; i < n_contacts_; i++) {
    multibody::GeomGeomCollider collider(plant_, contact_pairs_[i]);
    if (frictionless_ || n_friction_directions_ == 1)
      std::tie(phi_i, J_i) = collider.EvalPlanar(context_, planar_normal_);
    else
      std::tie(phi_i, J_i) =
          collider.EvalPolytope(context_, n_friction_directions_);

    // Signed distance value for contact i
    phi(i) = phi_i;

    // J_i is 3 x n_v_
    // row (0) is contact normal
    // rows (1-num_friction directions) are the contact tangents
    Jn.row(i) = J_i.row(0);
    if (frictionless_)
      continue;  // If frictionless_, we only need the normal force
    Jt.block(2 * i * n_friction_directions_, 0, 2 * n_friction_directions_,
             n_v_) = J_i.block(1, 0, 2 * n_friction_directions_, n_v_);
  }
}

std::pair<std::vector<VectorXd>, std::vector<VectorXd>>
LCSFactory::FindWitnessPoints() {
  std::vector<VectorXd> WCa;
  std::vector<VectorXd> WCb;

  for (int i = 0; i < n_contacts_; i++) {
    multibody::GeomGeomCollider collider(plant_, contact_pairs_[i]);
    auto [p_WCa, p_WCb] = collider.CalcWitnessPoints(context_);
    WCa.push_back(p_WCa);
    WCb.push_back(p_WCb);
  }

  return std::make_pair(WCa, WCb);
}

void LCSFactory::UpdateStateAndInput(
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  if (n_b_) {
    SetContext<double>(plant_, state, input.head(n_u_), input.tail(n_b_),
                       geoms_with_surface_velocity_, &context_);
  } else {
    SetContext<double>(plant_, state, input, &context_);
  }
  drake::VectorX<double> q_v_u(n_x_ + n_u_ + n_b_);
  q_v_u << state, input;
  drake::AutoDiffVecXd q_v_u_ad = drake::math::InitializeAutoDiff(q_v_u);
  SetPositionsAndVelocitiesIfNew<AutoDiffXd>(plant_ad_, q_v_u_ad.head(n_x_),
                                             &context_ad_);
  SetInputsIfNew<AutoDiffXd>(plant_ad_, q_v_u_ad(Eigen::seqN(n_x_, n_u_)),
                             &context_ad_);
  if (n_b_) {
    SetSurfaceVelocitiesIfNew<AutoDiffXd>(plant_ad_, q_v_u_ad.tail(n_b_),
                                          geoms_with_surface_velocity_,
                                          &context_ad_);
  }
}
// Linearizes the dynamics of a multibody plant_ into a Linear Complementarity
// System (LCS)
LCS LCSFactory::GenerateLCS() {
  if (!frictionless_) DRAKE_DEMAND(mu_.size() == (size_t)n_contacts_);

  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu_.data(), mu_.size());

  /*============== Formulate A, B and d Matrices ==================*/
  // Calculate mass matrix M(q)
  MatrixX<AutoDiffXd> M(n_v_, n_v_);
  plant_ad_.CalcMassMatrix(context_ad_, &M);

  // Calculate Coriolis term C(q, v)v
  AutoDiffVecXd C(n_v_);
  plant_ad_.CalcBiasTerm(context_ad_, &C);

  std::cout << "plant ad positions: " << plant_ad_.num_positions() << std::endl;
  std::cout << "plant ad velocities: " << plant_ad_.num_velocities() << std::endl;
  std::cout << "plant ad actuators: " << plant_ad_.num_actuators() << std::endl;

  // Calculate generalized forces τ(u) = Bu
  auto B_dyn_ad = plant_ad_.MakeActuationMatrix();
  std::cout << "B_dyn_ad size: " << B_dyn_ad.rows() << " x " << B_dyn_ad.cols() << std::endl;
  auto u_ad = plant_ad_.get_actuation_input_port().Eval(context_ad_);
  std::cout << "u_ad: " << u_ad.rows() << " x " << u_ad.cols() << std::endl;
  AutoDiffVecXd tau_u = B_dyn_ad * u_ad;

  // Calculate generalized forces due to gravity τ₍g₎
  AutoDiffVecXd tau_g = plant_ad_.CalcGravityGeneralizedForces(context_ad_);

  // Get forces applied to the plant_
  MultibodyForces<AutoDiffXd> f_app(plant_ad_);
  plant_ad_.CalcForceElementsContribution(context_ad_, &f_app);

  // f(q, v, u) =  M(q)⁻¹(τ(u) + τ₍g₎ + fₐₚₚ(q, v, u) - C(q, v))
  AutoDiffVecXd f_qvu =
      M.ldlt().solve(tau_g + tau_u + f_app.generalized_forces() - C);

  // f(q*, v*, u*)
  VectorXd f_qvu_nominal = ExtractValue(f_qvu);
  // Jacobian of f(q, v, u) w.r.t. q, v, u
  MatrixXd Jf = ExtractGradient(f_qvu);
  if (Jf.cols() != n_x_ + n_u_) {
    throw std::runtime_error(fmt::format(
        "Jacobian of f(q, v, u) has unexpected number of columns: {}. "
        "Expected: {} + {} = {}",
        Jf.cols(), n_x_, n_u_, n_x_ + n_u_));
  }

  VectorXd qvu_nominal(n_q_ + n_v_ + n_u_);
  qvu_nominal << plant_.GetPositions(context_), plant_.GetVelocities(context_),
      plant_.get_actuation_input_port().Eval(context_);
  VectorXd Jf_qvu_nominal = Jf * qvu_nominal;
  // dᵥ = f(q*, v*, u*) - Jf * (q*, v*, u*)
  VectorXd d_v = f_qvu_nominal - Jf_qvu_nominal;

  // State dependent mapping q̇ = N(q)v
  Eigen::SparseMatrix<double> Nqt;
  Nqt = plant_.MakeVelocityToQDotMap(context_);
  MatrixXd qdotNv = MatrixXd(Nqt);

  MatrixXd Jf_q = Jf.block(0, 0, n_v_, n_q_);
  MatrixXd Jf_v = Jf.block(0, n_q_, n_v_, n_v_);
  MatrixXd Jf_u = Jf.block(0, n_x_, n_v_, n_u_);

  // Matrices for contact-free dynamics
  MatrixXd A(n_x_, n_x_);
  MatrixXd B(n_x_, n_u_ + n_b_);
  VectorXd d(n_x_);

  // Formulate A matrix
  A.block(0, 0, n_q_, n_q_) =
      MatrixXd::Identity(n_q_, n_q_) + dt_ * dt_ * qdotNv * Jf_q;
  A.block(0, n_q_, n_q_, n_v_) = dt_ * qdotNv + dt_ * dt_ * qdotNv * Jf_v;
  A.block(n_q_, 0, n_v_, n_q_) = dt_ * Jf_q;
  A.block(n_q_, n_q_, n_v_, n_v_) = dt_ * Jf_v + MatrixXd::Identity(n_v_, n_v_);

  // Formulate B matrix
  B.block(0, 0, n_q_, n_u_) = dt_ * dt_ * qdotNv * Jf_u;
  B.block(n_q_, 0, n_v_, n_u_) = dt_ * Jf_u;
  if (n_b_) {
    B.block(0, n_u_, n_x_, n_b_) = MatrixXd::Zero(n_x_, n_b_);
  }
  // Formulate d vector
  d.head(n_q_) = dt_ * dt_ * qdotNv * d_v;
  d.tail(n_v_) = dt_ * d_v;

  /*============== Formulate A, B and d Matrices ==================*/
  /*============== Calculate Contact Jacobians ==================*/

  // State dependent inverse mapping v = N⁺(q)⋅q̇
  Eigen::SparseMatrix<double> NqI;
  NqI = plant_.MakeQDotToVelocityMap(context_);
  MatrixXd vNqdot = MatrixXd(NqI);

  VectorXd phi;  // Signed distance values for contacts
  MatrixXd Jn;   // Normal contact Jacobian
  MatrixXd Jt;   // Tangential contact Jacobian
  ComputeContactJacobian(phi, Jn, Jt);

  /*============== Calculate Contact Jacobians ==================*/
  /*============== Formulate D, E, F, H and c Matrices ==================*/

  // Matrices with contact variables
  MatrixXd D = MatrixXd::Zero(n_x_, n_lambda_);
  MatrixXd E = MatrixXd::Zero(n_lambda_, n_x_);
  MatrixXd F = MatrixXd::Zero(n_lambda_, n_lambda_);
  MatrixXd H = MatrixXd::Zero(n_lambda_, n_u_ + n_b_);
  VectorXd c = VectorXd::Zero(n_lambda_);

  if (contact_model_ == ContactModel::kStewartAndTrinkle) {
    FormulateStewartTrinkleContactDynamics(phi, Jn, Jt, Jf_q, Jf_v, Jf_u, d_v,
                                           vNqdot, qdotNv, muXd, M, D, E, F, H,
                                           c);
  } else if (contact_model_ == ContactModel::kAnitescu) {
    FormulateAnitescuContactDynamics(phi, Jn, Jt, Jf_q, Jf_v, Jf_u, d_v, vNqdot,
                                     qdotNv, muXd, M, D, E, F, H, c);
  } else if (contact_model_ ==
             ContactModel::kFrictionlessSpring) {  // Frictionless spring
    FormulateFrictionlessSpringContactDynamics(
        phi, Jn, qdotNv, options_.spring_stiffness, M, D, E, F, H, c);
  } else {
    throw std::out_of_range("Unsupported contact model.");
  }
  /*============== Formulate D, E, F, G and c Matrices ==================*/

  return LCS(A, B, D, d, E, F, H, c, options_.N, dt_);  // Return the system;
}

void LCSFactory::FormulateFrictionlessSpringContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& qdotNv,
    const double& spring_stiffness, MatrixX<AutoDiffXd>& M, MatrixXd& D,
    MatrixXd& E, MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(Jn.transpose());

  D.block(0, 0, n_q_, n_contacts_) = dt_ * dt_ * qdotNv * MinvJ_n_T;
  D.block(n_q_, 0, n_v_, n_contacts_) = dt_ * MinvJ_n_T;

  F = MatrixXd::Identity(n_contacts_, n_contacts_);

  c = spring_stiffness * phi;
}

void LCSFactory::FormulateStewartTrinkleContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& Jt,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
    const VectorXd& mu_, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
    MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_n_T = M_ldlt.solve(Jn.transpose());
  MatrixXd MinvJ_t_T = M_ldlt.solve(Jt.transpose());

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t =
      MatrixXd::Zero(n_contacts_, 2 * n_contacts_ * n_friction_directions_);
  for (int i = 0; i < n_contacts_; i++) {
    E_t.block(i, i * (2 * n_friction_directions_), 1,
              2 * n_friction_directions_) =
        MatrixXd::Ones(1, 2 * n_friction_directions_);
  }

  // Formulate D matrix (state-force)
  D.block(0, 2 * n_contacts_, n_q_, 2 * n_contacts_ * n_friction_directions_) =
      dt_ * dt_ * qdotNv * MinvJ_t_T;
  D.block(n_q_, 2 * n_contacts_, n_v_,
          2 * n_contacts_ * n_friction_directions_) = dt_ * MinvJ_t_T;
  D.block(0, n_contacts_, n_q_, n_contacts_) = dt_ * dt_ * qdotNv * MinvJ_n_T;
  D.block(n_q_, n_contacts_, n_v_, n_contacts_) = dt_ * MinvJ_n_T;

  // Formulate E matrix (force-state)
  E.block(n_contacts_, 0, n_contacts_, n_q_) =
      dt_ * dt_ * Jn * Jf_q + Jn * vNqdot;
  E.block(2 * n_contacts_, 0, 2 * n_contacts_ * n_friction_directions_, n_q_) =
      dt_ * Jt * Jf_q;
  E.block(n_contacts_, n_q_, n_contacts_, n_v_) =
      dt_ * Jn + dt_ * dt_ * Jn * Jf_v;
  E.block(2 * n_contacts_, n_q_, 2 * n_contacts_ * n_friction_directions_,
          n_v_) = Jt + dt_ * Jt * Jf_v;

  // Formulate F matrix (force-force)
  F.block(0, n_contacts_, n_contacts_, n_contacts_) = mu_.asDiagonal();
  F.block(0, 2 * n_contacts_, n_contacts_,
          2 * n_contacts_ * n_friction_directions_) = -E_t;
  F.block(n_contacts_, n_contacts_, n_contacts_, n_contacts_) =
      dt_ * dt_ * Jn * MinvJ_n_T;
  F.block(n_contacts_, 2 * n_contacts_, n_contacts_,
          2 * n_contacts_ * n_friction_directions_) =
      dt_ * dt_ * Jn * MinvJ_t_T;
  F.block(2 * n_contacts_, 0, 2 * n_contacts_ * n_friction_directions_,
          n_contacts_) = E_t.transpose();
  F.block(2 * n_contacts_, n_contacts_,
          2 * n_contacts_ * n_friction_directions_, n_contacts_) =
      dt_ * Jt * MinvJ_n_T;
  F.block(2 * n_contacts_, 2 * n_contacts_,
          2 * n_contacts_ * n_friction_directions_,
          2 * n_contacts_ * n_friction_directions_) = dt_ * Jt * MinvJ_t_T;

  // Formulate H matrix (force-input)
  H.block(n_contacts_, 0, n_contacts_, n_u_) = dt_ * dt_ * Jn * Jf_u;
  H.block(2 * n_contacts_, 0, 2 * n_contacts_ * n_friction_directions_, n_u_) =
      dt_ * Jt * Jf_u;
  if (n_b_) {
    // This project defines the contact frame C such that the contact normal is
    // along the +X axis. This is different from Drake's definition of the
    // surface normal at the contact point which is defined along the +Z axis.
    // R_C transforms a vector from drake's frame for the surface normal to the
    // contact frame used in this project.
    static const Eigen::Matrix3d R_C = Eigen::Matrix3d::Identity();
    // drake::math::RotationMatrixd::MakeYRotation(M_PI_2) *
    // drake::math::RotationMatrixd::MakeZRotation(M_PI_2);
    const Eigen::Matrix<double, Eigen::Dynamic, 3> fb = GetForceBasis();

    const auto& query_port =
        plant_.get_geometry_query_input_port()
            .Eval<drake::geometry::QueryObject<double>>(context_);

    for (int i = 0; i < n_contacts_; i++) {
      multibody::GeomGeomCollider collider(plant_, contact_pairs_[i]);
      const auto& query_result = collider.GetGeometryQueryResult(context_);
      const auto& [geom_a, geom_b] = contact_pairs_[i];
      // Contract frame as seen from world frame
      const Eigen::Matrix3d R_WC =
          drake::math::RotationMatrixd::MakeFromOneVector(
              query_result.signed_distance_pair.nhat_BA_W, 0)
              .matrix()
              .transpose();

      if (auto iter = geoms_with_surface_velocity_.find(geom_a);
          iter != geoms_with_surface_velocity_.end()) {
        // Pose of geometry in world frame
        const Eigen::Matrix3d& X_WG =
            query_port.GetPoseInWorld(geom_a).rotation().matrix();

        int idx = std::distance(geoms_with_surface_velocity_.begin(), iter);
        Eigen::Vector3d sv =
            plant_
                .GetSurfaceVelocity(context_, geom_a, inspector_,
                                    drake::math::RigidTransformd::Identity(),
                                    query_result.signed_distance_pair.p_ACa)
                .normalized();
        double n_J_a = fb.row(0) * R_WC * R_C * X_WG * sv;
        Eigen::VectorXd t_J_a = fb.block(1, 0, 2 * n_friction_directions_, 3) *
                                R_WC * R_C * X_WG * sv;
        H(n_contacts_ + i, n_u_ + idx) = n_J_a;
        H.block(2 * n_contacts_ + i * 2 * n_friction_directions_, n_u_ + idx, 2 * n_friction_directions_,
                1) = t_J_a;
      }
      if (auto iter = geoms_with_surface_velocity_.find(geom_b);
          iter != geoms_with_surface_velocity_.end()) {
        int idx = std::distance(geoms_with_surface_velocity_.begin(), iter);
        // Pose of geometry in world frame
        const Eigen::Matrix3d& X_WG =
            query_port.GetPoseInWorld(geom_b).rotation().matrix();

        Eigen::Vector3d sv =
            plant_
                .GetSurfaceVelocity(context_, geom_b, inspector_,
                                    drake::math::RigidTransformd::Identity(),
                                    query_result.signed_distance_pair.p_BCb)
                .normalized();
        double n_J_b = fb.row(0) * R_WC * R_C * X_WG * sv;
        Eigen::VectorXd t_J_b = fb.block(1, 0, 2 * n_friction_directions_, 3) *
                                R_WC * R_C * X_WG * sv;
        H(n_contacts_ + i, n_u_ + idx) = -n_J_b;
        H.block(2 * n_contacts_ + i, n_u_ + idx, 2 * n_friction_directions_,
                1) = -t_J_b;
      }
    }
  }

  // Formulate c vector
  c.segment(n_contacts_, n_contacts_) =
      phi + dt_ * dt_ * Jn * d_v - Jn * vNqdot * plant_.GetPositions(context_);
  c.segment(2 * n_contacts_, 2 * n_contacts_ * n_friction_directions_) =
      Jt * dt_ * d_v;
}

void LCSFactory::FormulateAnitescuContactDynamics(
    const VectorXd& phi, const MatrixXd& Jn, const MatrixXd& Jt,
    const MatrixXd& Jf_q, const MatrixXd& Jf_v, const MatrixXd& Jf_u,
    const VectorXd& d_v, const MatrixXd& vNqdot, const MatrixXd& qdotNv,
    const VectorXd& mu_, MatrixX<AutoDiffXd>& M, MatrixXd& D, MatrixXd& E,
    MatrixXd& F, MatrixXd& H, VectorXd& c) {
  auto M_ldlt = ExtractValue(M).ldlt();

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  // (ne) number of directions in firctional cone
  MatrixXd E_t =
      MatrixXd::Zero(n_contacts_, 2 * n_contacts_ * n_friction_directions_);
  for (int i = 0; i < n_contacts_; i++) {
    E_t.block(i, i * (2 * n_friction_directions_), 1,
              2 * n_friction_directions_) =
        MatrixXd::Ones(1, 2 * n_friction_directions_);
  }

  // Apply same friction coefficients to each friction direction for same
  // contact
  VectorXd anitescu_mu_vec = VectorXd::Zero(n_lambda_);
  for (int i = 0; i < mu_.rows(); i++) {
    anitescu_mu_vec.segment((2 * n_friction_directions_) * i,
                            2 * n_friction_directions_) =
        mu_(i) * VectorXd::Ones(2 * n_friction_directions_);
  }
  MatrixXd anitescu_mu_matrix = anitescu_mu_vec.asDiagonal();

  // Constructing friction bases Jc = EᵀJₙ + μJₜ
  MatrixXd J_c = E_t.transpose() * Jn + anitescu_mu_matrix * Jt;

  MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

  // Formulate D matrix (state-force)
  D.block(0, 0, n_q_, n_lambda_) = dt_ * dt_ * qdotNv * MinvJ_c_T;
  D.block(n_q_, 0, n_v_, n_lambda_) = dt_ * MinvJ_c_T;

  // Formulate E matrix (force-state)
  E.block(0, 0, n_lambda_, n_q_) =
      dt_ * J_c * Jf_q + E_t.transpose() * Jn * vNqdot / dt_;
  E.block(0, n_q_, n_lambda_, n_v_) = J_c + dt_ * J_c * Jf_v;

  // Formulate F matrix (force-force)
  F = dt_ * J_c * MinvJ_c_T;

  // Formulate H matrix (force-input)
  H.block(0, 0, 2 * n_contacts_ * n_friction_directions_, n_u_) =
      dt_ * J_c * Jf_u;

  if (n_b_) {
    // C3 defines the contact frame such that the contact normal is along the +X
    // axis. This is different from Drake's definition that a shape normal
    // exists along the +Z axis at the point where the normal is queried. R_WC
    // transforms a vector using drake's frame to C3's frame.
    // TODO(@juan): Check if R_WC is necessary or not
    static const Eigen::Matrix3d R_C = Eigen::Matrix3d::Identity();
    // drake::math::RotationMatrixd::MakeYRotation(M_PI_2) *
    // drake::math::RotationMatrixd::MakeZRotation(M_PI_2);
    const Eigen::Vector3d Ek =
        Eigen::Vector3d::Ones(2 * n_friction_directions_);
    const Eigen::Matrix<double, Eigen::Dynamic, 3> fb = GetForceBasis();

    const auto& query_port =
        plant_.get_geometry_query_input_port()
            .Eval<drake::geometry::QueryObject<double>>(context_);

    // Loop through all contact pairs, and add surface velocity jacobian
    // for each geometry with corresponding parameters
    for (int i = 0; i < n_contacts_; i++) {
      // Get geometry query result to access witness points in their geometry
      // frame
      multibody::GeomGeomCollider collider(plant_, contact_pairs_[i]);
      const auto query_result = collider.GetGeometryQueryResult(context_);
      // Contract frame as seen from world frame
      const Eigen::Matrix3d R_WC =
          drake::math::RotationMatrixd::MakeFromOneVector(
              query_result.signed_distance_pair.nhat_BA_W, 0)
              .matrix()
              .transpose();

      // Loop through contact geometries and add surface velocity jacobians.
      // All this happens in the contact frame.
      const auto& [geom_a, geom_b] = contact_pairs_[i];
      if (auto iter = geoms_with_surface_velocity_.find(geom_a);
          iter != geoms_with_surface_velocity_.end()) {
        // Index of the input element corresponding to geom_a in the input
        // vector u.
        int idx = std::distance(geoms_with_surface_velocity_.begin(), iter);

        // Pose of geometry in world frame
        const Eigen::Matrix3d& X_WG =
            query_port.GetPoseInWorld(geom_a).rotation().matrix();

        // Query surface velocity and normalize to obtain the surface velocity
        // vector only.
        Eigen::Vector3d sv =
            plant_
                .GetSurfaceVelocity(context_, geom_a, inspector_,
                                    drake::math::RigidTransformd::Identity(),
                                    query_result.signed_distance_pair.p_ACa)
                .normalized();
        // Build jacobians and add to control matrix H.
        Eigen::VectorXd n_J_a = Ek * fb.row(0) * R_WC * R_C * X_WG * sv;
        Eigen::VectorXd t_J_a = fb.block(1, 0, 2 * n_friction_directions_, 3) *
                                R_WC * R_C * X_WG * sv;
        H.block(i * 2 * n_friction_directions_, n_u_ + idx,
                2 * n_friction_directions_, 1) = n_J_a + t_J_a;
      }
      // Same for geom_b
      if (auto iter = geoms_with_surface_velocity_.find(geom_b);
          iter != geoms_with_surface_velocity_.end()) {
        int idx = std::distance(geoms_with_surface_velocity_.begin(), iter);
        // Pose of geometry in world frame
        // Pose of geometry in world frame
        const Eigen::Matrix3d& X_WG =
            query_port.GetPoseInWorld(geom_b).rotation().matrix();
        Eigen::Vector3d sv =
            plant_
                .GetSurfaceVelocity(context_, geom_b, inspector_,
                                    drake::math::RigidTransformd::Identity(),
                                    query_result.signed_distance_pair.p_BCb)
                .normalized();
        Eigen::VectorXd n_J_b = Ek * fb.row(0) * R_WC * R_C * X_WG * sv;
        Eigen::VectorXd t_J_b = fb.block(1, 0, 2 * n_friction_directions_, 3) *
                                R_WC * R_C * X_WG * sv;
        H.block(i * 2 * n_friction_directions_, n_u_ + idx,
                2 * n_friction_directions_, 1) = -(n_J_b + t_J_b);
      }
    }
  }

  // Formulate c vector
  c = E_t.transpose() * phi / dt_ + dt_ * J_c * d_v -
      E_t.transpose() * Jn * vNqdot * plant_.GetPositions(context_) / dt_;
}

LCS LCSFactory::LinearizePlantToLCS(
    const drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>& context,
    const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>& context_ad,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms,
    const LCSFactoryOptions& options,
    const Eigen::Ref<const drake::VectorX<double>>& state,
    const Eigen::Ref<const drake::VectorX<double>>& input) {
  LCSFactory lcs_factory(plant, context, plant_ad, context_ad, contact_geoms,
                         options);
  lcs_factory.UpdateStateAndInput(state, input);
  return lcs_factory.GenerateLCS();
}

std::pair<MatrixXd, std::vector<VectorXd>>
LCSFactory::GetContactJacobianAndPoints() {
  VectorXd phi;  // Signed distance values for contacts
  MatrixXd Jn;   // Normal contact Jacobian
  MatrixXd Jt;   // Tangential contact Jacobian
  ComputeContactJacobian(phi, Jn, Jt);
  auto [_, contact_points] = FindWitnessPoints();

  if (frictionless_) {
    // if frictionless_, we only need the normal jacobian
    return std::make_pair(Jn, contact_points);
  }

  if (contact_model_ == ContactModel::kStewartAndTrinkle) {
    // if Stewart and Trinkle model, concatenate the normal and tangential
    // jacobian
    MatrixXd J_c = MatrixXd::Zero(
        n_contacts_ + 2 * n_contacts_ * n_friction_directions_, n_v_);
    J_c << Jn, Jt;
    return std::make_pair(J_c, contact_points);
  }

  // Model is Anitescu
  int n_lambda_ = 2 * n_contacts_ * n_friction_directions_;

  // Eₜ = blkdiag(e,.., e), e ∈ 1ⁿᵉ
  MatrixXd E_t = MatrixXd::Zero(n_contacts_, n_lambda_);
  for (int i = 0; i < n_contacts_; i++) {
    E_t.block(i, i * (2 * n_friction_directions_), 1,
              2 * n_friction_directions_) =
        MatrixXd::Ones(1, 2 * n_friction_directions_);
  }

  // Apply same friction coefficients to each friction direction
  // of the same contact
  if (!frictionless_) DRAKE_DEMAND(mu_.size() == (size_t)n_contacts_);
  VectorXd muXd =
      Eigen::Map<const VectorXd, Eigen::Unaligned>(mu_.data(), mu_.size());

  VectorXd mu_vector = VectorXd::Zero(n_lambda_);
  for (int i = 0; i < muXd.rows(); i++) {
    mu_vector.segment((2 * n_friction_directions_) * i,
                      2 * n_friction_directions_) =
        muXd(i) * VectorXd::Ones(2 * n_friction_directions_);
  }
  MatrixXd mu_matrix = mu_vector.asDiagonal();

  // Constructing friction bases  Jc = EᵀJₙ + μJₜ
  MatrixXd J_c = E_t.transpose() * Jn + mu_matrix * Jt;
  return std::make_pair(J_c, contact_points);
}

LCS LCSFactory::FixSomeModes(const LCS& other, set<int> active_lambda_inds,
                             set<int> inactive_lambda_inds) {
  std::vector<int> remaining_inds;

  // Assumes constant number of contacts per index
  int n_lambda_ = other.F()[0].rows();

  // Need to solve for lambda_active in terms of remaining elements
  // Build temporary [F1, F2] by eliminating rows for inactive
  for (int i = 0; i < n_lambda_; i++) {
    // active/inactive must be exclusive
    DRAKE_ASSERT(!active_lambda_inds.count(i) ||
                 !inactive_lambda_inds.count(i));

    // In C++20, could use contains instead of count
    if (!active_lambda_inds.count(i) && !inactive_lambda_inds.count(i)) {
      remaining_inds.push_back(i);
    }
  }

  int n_remaining = remaining_inds.size();
  int n_active = active_lambda_inds.size();

  std::vector<MatrixXd> A, B, D, E, F, H;
  std::vector<VectorXd> d, c;

  // Build selection matrices:
  // S_a selects active indices
  // S_r selects remaining indices

  MatrixXd S_a = MatrixXd::Zero(n_active, n_lambda_);
  MatrixXd S_r = MatrixXd::Zero(n_remaining, n_lambda_);

  for (int i = 0; i < n_remaining; i++) {
    S_r(i, remaining_inds[i]) = 1;
  }
  {
    int i = 0;
    for (auto ind_j : active_lambda_inds) {
      S_a(i, ind_j) = 1;
      i++;
    }
  }

  for (int k = 0; k < other.N(); k++) {
    Eigen::BDCSVD<MatrixXd> svd;
    svd.setThreshold(1e-5);
    svd.compute(S_a * other.F()[k] * S_a.transpose(),
                Eigen::ComputeFullU | Eigen::ComputeFullV);

    // F_active likely to be low-rank due to friction, but that should be OK
    // MatrixXd res = svd.solve(F_ar);

    // Build new complementarity constraints
    // F_a_inv = pinv(S_a * F * S_a^T)
    // 0 <= \lambda_k \perp E_k x_k + F_k \lambda_k + H_k u_k + c_k
    // 0 = S_a *(E x + F S_a^T \lambda_a + F S_r^T \lambda_r + H_k u_k + c_k)
    // \lambda_a = -F_a_inv * (S_a F S_r^T * lambda_r + S_a E x + S_a H u +
    // S_a c)
    //
    // 0 <= \lambda_r \perp S_r (I - F S_a^T F_a_inv S_a) E x + ...
    //                      S_r (I - F S_a^T F_a_inv S_a) F S_r^T \lambda_r +
    //                      ... S_r (I - F S_a^T F_a_inv S_a) H u + ... S_r(I
    //                      - F S_a^T F_a_inv S_a) c
    //
    // Calling L = S_r (I - F S_a^T F_a_inv S_a)S_r * other.D()[k]
    //  E_k = L E
    //  F_k = L F S_r^t
    //  H_k = L H
    //  c_k = L c
    MatrixXd L = S_r * (MatrixXd::Identity(n_lambda_, n_lambda_) -
                        other.F()[k] * S_a.transpose() * svd.solve(S_a));
    MatrixXd E_k = L * other.E()[k];
    MatrixXd F_k = L * other.F()[k] * S_r.transpose();
    MatrixXd H_k = L * other.H()[k];
    MatrixXd c_k = L * other.c()[k];

    // Similarly,
    //  A_k = A - D * S_a^T * F_a_inv * S_a * E
    //  B_k = B - D * S_a^T * F_a_inv * S_a * H
    //  D_k = D * S_r^T - D * S_a^T  * F_a_inv * S_a F S_r^T
    //  d_k = d - D * S_a^T F_a_inv * S_a * c
    //
    //  Calling P = D * S_a^T * F_a_inv * S_a
    //
    //  A_k = A - P E
    //  B_k = B - P H
    //  D_k = S_r D - P S_r^T
    //  d_k = d - P c
    MatrixXd P = other.D()[k] * S_a.transpose() * svd.solve(S_a);
    MatrixXd A_k = other.A()[k] - P * other.E()[k];
    MatrixXd B_k = other.B()[k] - P * other.H()[k];
    MatrixXd D_k = other.D()[k] * S_r.transpose() - P * S_r.transpose();
    MatrixXd d_k = other.d()[k] - P * other.c()[k];
    E.push_back(E_k);
    F.push_back(F_k);
    H.push_back(H_k);
    c.push_back(c_k);
    A.push_back(A_k);
    B.push_back(B_k);
    D.push_back(D_k);
    d.push_back(d_k);
  }
  return LCS(A, B, D, d, E, F, H, c, other.dt());
}

int LCSFactory::GetNumContactVariables(ContactModel contact_model,
                                       int num_contacts,
                                       int num_friction_directions) {
  if (contact_model == ContactModel::kFrictionlessSpring) {
    return num_contacts;  // Only normal forces
  } else if (contact_model == ContactModel::kStewartAndTrinkle) {
    return 2 * num_contacts +
           2 * num_contacts *
               num_friction_directions;  // Compute contact variable count
                                         // for Stewart-Trinkle model
  } else if (contact_model == ContactModel::kAnitescu) {
    return 2 * num_contacts *
           num_friction_directions;  // Compute contact variable
                                     // count for Anitescu model
  }
  throw std::out_of_range("Unknown contact model.");
}

int LCSFactory::GetNumContactVariables(const LCSFactoryOptions options) {
  multibody::ContactModel contact_model =
      GetContactModelMap().at(options.contact_model);
  return GetNumContactVariables(contact_model, options.num_contacts,
                                options.num_friction_directions);
}

int LCSFactory::GetNumContactVelocityBiases(
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context,
    const std::vector<drake::SortedPair<drake::geometry::GeometryId>>&
        contact_geoms) {
  int n_b = 0;
  const auto& inspector = plant.EvalSceneGraphInspector(context);
  std::set<drake::geometry::GeometryId> geoms_with_surface_params;
  // Loop through the contact geometries and increase count if they
  // have surface velocity proximity parameters
  for (const auto& [geom_a, geom_b] : contact_geoms) {
    std::optional<std::pair<double, Vector3<double>>> surface_params =
        plant.GetCurrentSurfaceSpeedAndNormal(context, geom_a, inspector);
    if (surface_params.has_value() && (geoms_with_surface_params.find(geom_a) ==
                                       geoms_with_surface_params.end())) {
      n_b++;
      geoms_with_surface_params.insert(geom_a);
    }
    surface_params =
        plant.GetCurrentSurfaceSpeedAndNormal(context, geom_b, inspector);
    if (surface_params.has_value() && (geoms_with_surface_params.find(geom_b) ==
                                       geoms_with_surface_params.end())) {
      n_b++;
      geoms_with_surface_params.insert(geom_b);
    }
  }
  return n_b;
}

void LCSFactory::ComputeSetOfGeometriesWithSurfaceVelocity() {
  const auto& inspector = plant_.EvalSceneGraphInspector(context_);
  // Loop through the contact geometries and insert only those with
  // surface velocity parameters
  for (const auto& [geom_a, geom_b] : contact_pairs_) {
    std::optional<std::pair<double, Vector3<double>>> surface_params =
        plant_.GetCurrentSurfaceSpeedAndNormal(context_, geom_a, inspector);
    if (surface_params.has_value()) {
      geoms_with_surface_velocity_.insert(geom_a);
    }
    surface_params =
        plant_.GetCurrentSurfaceSpeedAndNormal(context_, geom_b, inspector);
    if (surface_params.has_value()) {
      geoms_with_surface_velocity_.insert(geom_b);
    }
  }
}

std::set<drake::geometry::GeometryId> GetSetOfGeometriesWithSurfaceVelocity(
    const LCSFactory& lcsf) {
  return lcsf.geoms_with_surface_velocity_;
}

Eigen::Matrix<double, Eigen::Dynamic, 3> LCSFactory::GetForceBasis() const {
  if (frictionless_ || n_friction_directions_ == 1) {
    // TODO(@juan) implement computer planar forces using contact normal
    return GeomGeomCollider<double>::ComputePlanarForceBasis(
        Eigen::Vector3d(3, 1, 1).normalized(), planar_normal_);
  }
  return GeomGeomCollider<double>::ComputePolytopeForceBasis(
      n_friction_directions_);
}

const drake::geometry::SceneGraphInspector<double>&
LCSFactory::getSceneGraphInspector(const MultibodyPlant<double>& plant,
                                   const Context<double>& context) {
  return plant.EvalSceneGraphInspector(context);
}

}  // namespace multibody
}  // namespace c3