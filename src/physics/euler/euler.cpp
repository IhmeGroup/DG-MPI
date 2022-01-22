#include "physics/euler/euler.h"

namespace Physics {

template<int dim> int Euler<dim>::get_NS(){
    return dim + 2;
}

template<int dim> inline
void Euler<dim>::set_physical_params(rtype GasConstant, 
    rtype SpecificHeatRatio){

    R = GasConstant;
    gamma = SpecificHeatRatio;
}

template<int dim> DG_KOKKOS_FUNCTION
rtype Euler<dim>::get_pressure(Kokkos::View<const rtype*> U) {
    // unpack
    auto mom = Kokkos::subview(U, Kokkos::make_pair(1, dim + 1));
    const rtype rKE = KokkosBlas::dot(mom, mom) / U(0);
    return (gamma - 1.) * (U(dim + 1) - 0.5 * rKE);
}

template<> DG_KOKKOS_FUNCTION
void Euler<2>::conv_flux_physical(
    Kokkos::View<const rtype*> U,
    Kokkos::View<rtype**> F){

    const rtype P = get_pressure(U);
    const rtype r1 = 1. / U(0);
    const rtype ru = U(1);
    const rtype rv = U(2);
    const rtype rE = U(3);

    F(0, 0) = ru;
    F(1, 0) = ru * ru * r1 + P;
    F(2, 0) = ru * rv * r1;
    F(3, 0) = (rE + P) * ru * r1;

    F(0, 1) = rv;
    F(1, 1) = ru * rv * r1;
    F(2, 1) = rv * rv * r1 + P;
    F(3, 1) = (rE + P) * rv * r1;
}

template<> DG_KOKKOS_FUNCTION
void Euler<3>::conv_flux_physical(
    Kokkos::View<const rtype*> U,
    Kokkos::View<rtype**> F){

    const rtype P = get_pressure(U);
    const rtype r1 = 1. / U(0);
    const rtype ru = U(1);
    const rtype rv = U(2);
    const rtype rw = U(3);
    const rtype rE = U(4);

    F(0, 0) = ru;
    F(1, 0) = ru * ru * r1 + P;
    F(2, 0) = ru * rv * r1;
    F(3, 0) = ru * rw * r1;
    F(4, 0) = (rE + P) * ru * r1;

    F(0, 1) = rv;
    F(1, 1) = rv * ru * r1;
    F(2, 1) = rv * rv * r1 + P;
    F(3, 1) = rv * rw * r1;
    F(4, 1) = (rE + P) * rv * r1;

    F(0, 2) = rw;
    F(1, 2) = rw * ru * r1;
    F(2, 2) = rw * rv * r1;
    F(3, 2) = rw * rw * r1 + P;
    F(4, 2) = (rE + P) * rw * r1;
}


// template<> inline
// void EulerBase<2>::conv_flux_normal(const rtype *U, const rtype P, const rtype *N, rtype *F) {
//     const rtype r = U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rE = U[3];
//     const rtype un = (ru * N[0] + rv * N[1]) / r;

//     F[0] = r * un;
//     F[1] = ru * un + P * N[0];
//     F[2] = rv * un + P * N[1];
//     F[3] = (rE + P) * un;
// }

// template<> inline
// void EulerBase<3>::conv_flux_normal(const rtype *U, const rtype P, const rtype *N, rtype *F) {
//     const rtype r = U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rw = U[3];
//     const rtype rE = U[4];
//     const rtype un = Math::dot<3>(U + 1, N) / r;

//     F[0] = r * un;
//     F[1] = ru * un + P * N[0];
//     F[2] = rv * un + P * N[1];
//     F[3] = rw * un + P * N[2];
//     F[4] = (rE + P) * un;
// }


/*====================================================*/
/* STATIC METHODS IMPLEMENTATIONS FOR EulerBase<dim>. */
/*====================================================*/

// template<unsigned dim> DG_KOKKOS_INLINE_FUNCTION
// void EulerBase<dim>::vol_fluxes(
//     const EquationInput &input,
//     const Physics::IdealGasParams &params,
//     const rtype *U,
//     const rtype *gU,
//     rtype *F,
//     rtype *gF) {

//     // source terms are not taken into account

//     EulerBase<dim>::Fc(params, U, gF);
// }

// template<> DG_KOKKOS_INLINE_FUNCTION
// void EulerBase<2>::Fc(
//     const Physics::IdealGasParams &params,
//     const rtype *U,
//     rtype *gF) {

//     const rtype r1 = 1. / U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rE = U[3];
//     const rtype P = 100000.0;//Physics::IdealGas<2>::pressure(params, U);

//     gF[0] = ru;
//     gF[1] = ru*ru*r1 + P;
//     gF[2] = ru*rv*r1;
//     gF[3] = (rE + P)*ru*r1;

//     gF[4] = rv;
//     gF[5] = ru*rv*r1;
//     gF[6] = rv*rv*r1 + P;
//     gF[7] = (rE + P)*rv*r1;
// }

// template<> DG_KOKKOS_INLINE_FUNCTION
// void EulerBase<3>::Fc(
//     const Physics::IdealGasParams &params,
//     const rtype *U,
//     rtype *gF) {

//     const rtype r1 = 1. / U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rw = U[3];
//     const rtype rE = U[4];
//     const rtype P = 100000.0;//Physics::IdealGas<3>::pressure(params, U);

//     gF[0] = ru;
//     gF[1] = ru*ru*r1 + P;
//     gF[2] = ru*rv*r1;
//     gF[3] = ru*rw*r1;
//     gF[4] = (rE + P)*ru*r1;

//     gF[5] = rv;
//     gF[6] = rv*ru*r1;
//     gF[7] = rv*rv*r1 + P;
//     gF[8] = rv*rw*r1;
//     gF[9] = (rE + P)*rv*r1;

//     gF[10] = rw;
//     gF[11] = rw*ru*r1;
//     gF[12] = rw*rv*r1;
//     gF[13] = rw*rw*r1 + P;
//     gF[14] = (rE + P)*rw*r1;
// }

// //--------------------------------------------------------------------------------------------------
// //--------------------------------------------------------------------------------------------------


// template<> inline
// void EulerBase<2>::analytic_state(
//     const EquationInput &input,
//     const AnalyticType type, rtype *U) const {

//     switch (type) {
//         case AnalyticType::Uniform:
//             AnalyticExpression::navier_stokes_uniform(&phys, input.ex_params, U);
//             break;
//         case AnalyticType::UniformPressure:
//             AnalyticExpression::navier_stokes_uniform_pressure(&phys, input.ex_params, U);
//             break;
//         case AnalyticType::IsentropicVortex:
//             AnalyticExpression::isentropic_vortex<2>(&phys, input.X, U);
//             break;
//         case AnalyticType::LeavingVortex:
//             AnalyticExpression::leaving_vortex<2>(&phys, input.X, input.t, U);
//             break;
//         case AnalyticType::Bump:
//             AnalyticExpression::isentropic_bump<2>(U);
//             break;
//         case AnalyticType::LinearProfile:
//             AnalyticExpression::linear_profile<2>(&(this->phys), input.X, U);
//             break;
//         default : {
//             throw InputException(
//                 "Wrong combination of equation and analytic type in EulerBase<2>.");
//         }
//     }
// }

// template<> inline
// void EulerBase<3>::analytic_state(
//     const EquationInput &input,
//     const AnalyticType type, rtype *U) const {

//     switch (type) {
//         case AnalyticType::Uniform:
//             AnalyticExpression::navier_stokes_uniform(&phys, input.ex_params, U);
//             break;
//         case AnalyticType::UniformPressure:
//             AnalyticExpression::navier_stokes_uniform_pressure(&phys, input.ex_params, U);
//             break;
//         case AnalyticType::IsentropicVortex:
//             AnalyticExpression::isentropic_vortex<3>(&phys, input.X, U);
//             break;
//         case AnalyticType::IsentropicVortexRotated:
//             AnalyticExpression::rotated_isentropic_vortex<3>(&phys, input.ex_params, input.X, U);
//             break;
//         case AnalyticType::Bump:
//             AnalyticExpression::isentropic_bump<3>(U);
//             break;
//         case AnalyticType::LinearProfile:
//             AnalyticExpression::linear_profile<3>(&(this->phys), input.X, U);
//             break;
//         default : {
//             throw InputException(
//                 "Wrong combination of equation and initialization type in EulerBase<3>.");
//         }
//     }
// }

// template<unsigned dim>
// rtype EulerBase<dim>::get_maximum_CFL(
//     const EquationInput &input,
//     const unsigned ne,
//     const rtype *Ue) const {

//     rtype CFL = -1.;
//     for (unsigned ie = 0; ie < ne; ie++) {
//         const rtype *U = Ue + ie * ns; // assumed layout
//         const rtype V = phys.comp_velocity_norm(U);
//         phys.set_state(U);
//         const rtype sos = phys.get_sos();
//         const rtype curr_CFL = input.dt / input.hL * (V + sos);
//         CFL = std::max(CFL, curr_CFL);
//     }
//     return CFL;
// }

// template<unsigned dim>
// void EulerBase<dim>::get_var(
//     const EquationInput &input,
//     const std::string var_name,
//     const unsigned ne,
//     const rtype *Ue,
//     const rtype *gUe,
//     rtype *v) const {

//     for (unsigned i = 0; i < ne; i++) {
//         // assumed layouts
//         const rtype *U = Ue + i * ns;
//         const rtype *dUdx = gUe + i * ns + 0 * ne * ns;
//         const rtype *dUdy = gUe + i * ns + 1 * ne * ns;
//         const rtype *dUdz = gUe + i * ns + 2 * ne * ns;
//         phys.set_state(U);
//         v[i] = phys.get_var(var_name, U, dUdx, dUdy, dUdz);
//     }
// }

// template<unsigned dim>
// void EulerBase<dim>::get_var(
//     const EquationInput &input,
//     const PhysicalVariable var,
//     const unsigned ne,
//     const rtype *Ue,
//     const rtype *gUe,
//     rtype *v) const {

//     for (unsigned i = 0; i < ne; i++) {
//         // assumed layouts
//         const rtype *U = Ue + i * ns;
//         const rtype *dUdx = gUe + i * ns + 0 * ne * ns;
//         const rtype *dUdy = gUe + i * ns + 1 * ne * ns;
//         const rtype *dUdz = gUe + i * ns + 2 * ne * ns;
//         phys.set_state(U);
//         v[i] = phys.get_var(var, U, dUdx, dUdy, dUdz);
//     }
// }

// template<unsigned dim>
// void EulerBase<dim>::get_vol_fluxes(
//     const EquationInput &input,
//     const rtype *U,
//     const rtype *gU,
//     rtype *F,
//     rtype *gF) {

//     const rtype P = phys.comp_P(U);
//     conv_flux_physical(input.stride_dim, U, P, gF);
// }

// template<unsigned dim>
// void EulerBase<dim>::get_weakprescribedpressureoutlet_boundary_fluxes(
//     const EquationInput &input,
//     const BoundaryData &bdata,
//     const rtype *U,
//     const rtype *gU,
//     const rtype *N,
//     rtype *F,
//     rtype *gF) {

//     assert(false);

//     // // get true normalized normal
//     // rtype n[dim];
//     // rtype N_norm = sqrt(Math::dot<dim>(N, N));
//     // for (unsigned idim=0; idim<dim; idim++) {
//     //     n[idim] = N[idim] / N_norm;
//     // }
//     //
//     // // interior quantities
//     // rtype rho = U[0];
//     // rtype un = Math::dot<dim>(U+1, n) / rho;
//     // rtype p = phys.comp_P(U);
//     // rtype c = phys.comp_sos_from_rhoP(rho, p);
//     // rtype gamma = phys.get_gamma();
//     //
//     // // boundary quantities
//     // assert(bdata.has_data);
//     // rtype pb = bdata.data[0];
//     // rtype rhob = rho * pow(pb/p, 1./gamma);
//     // rtype cb = phys.comp_sos_from_rhoP(rhob, pb);
//     // rtype unb = un + 2./(gamma-1.) * (c-cb);
//     //
//     // rtype Ub[NS];
//     // Ub[0] = rhob;
//     // for (unsigned i=0; i<dim; i++) {
//     //     Ub[1+i] = rhob*(unb-un)*n[i] + rhob/rho * U[1+i];
//     // }
//     // Ub[1+dim] = phys.comp_rhoE_from_UP(Ub, pb);
//     //
//     // rtype Fcn[NS];
//     // const rtype Pb = phys.comp_P(Ub);
//     // conv_flux_normal(Ub, Pb, N, Fcn);
//     // Math::cA_to_B(NS, -1, Fcn, F);
// }

// //--------------------------------------------------------------------------------------------------
// //--------------------------------------------------------------------------------------------------

// template<> inline
// void EulerBase<2>::get_Fc(const rtype *U, rtype *gF) {

//     const rtype r  = U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rE = U[3];
//     const rtype P  = phys.comp_P(U);
//     const rtype r1 = 1./r;

//     gF[0] = ru;
//     gF[1] = ru*ru*r1 + P;
//     gF[2] = ru*rv*r1;
//     gF[3] = (rE+P)*ru*r1;

//     gF[4] = rv;
//     gF[5] = ru*rv*r1;
//     gF[6] = rv*rv*r1 + P;
//     gF[7] = (rE+P)*rv*r1;
// }

// template<> inline
// void EulerBase<3>::get_Fc(const rtype *U, rtype *gF) {

//     const rtype r1 = 1. / U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rw = U[3];
//     const rtype rE = U[4];
//     const rtype P = phys.get_P();

//     gF[0] = ru;
//     gF[1] = ru * ru * r1 + P;
//     gF[2] = ru * rv * r1;
//     gF[3] = ru * rw * r1;
//     gF[4] = (rE + P) * ru * r1;

//     gF[5] = rv;
//     gF[6] = rv * ru * r1;
//     gF[7] = rv * rv * r1 + P;
//     gF[8] = rv * rw * r1;
//     gF[9] = (rE + P) * rv * r1;

//     gF[10] = rw;
//     gF[11] = rw * ru * r1;
//     gF[12] = rw * rv * r1;
//     gF[13] = rw * rw * r1 + P;
//     gF[14] = (rE + P) * rw * r1;
// }

// template<> inline
// void EulerBase<2>::get_Fcn(
//     const unsigned stride_state,
//     const rtype *U,
//     const rtype *N,
//     rtype *F) {

//     const rtype r = U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rE = U[3];
//     const rtype P = phys.get_P();
//     const rtype un = (ru*N[0] + rv*N[1]) / r;

//     F[0] = r * un;
//     F[1] = ru*un + P*N[0];
//     F[2] = rv*un + P*N[1];
//     F[3] = (rE + P) * un;
// }

// template<> inline
// void EulerBase<3>::get_Fcn(
//     const unsigned stride_state,
//     const rtype *U,
//     const rtype *N,
//     rtype *F) {

//     const rtype r = U[0];
//     const rtype ru = U[1];
//     const rtype rv = U[2];
//     const rtype rw = U[3];
//     const rtype rE = U[4];
//     const rtype P = phys.get_P();
//     const rtype un = Math::HandWritten::dot<3>(U+1, N) / r;

//     F[0] = r * un;
//     F[1] = ru*un + P*N[0];
//     F[2] = rv*un + P*N[1];
//     F[3] = rw*un + P*N[2];
//     F[4] = (rE + P) * un;
// }

// //--------------------------------------------------------------------------------------------------
// //--------------------------------------------------------------------------------------------------

// template<unsigned dim, InviscNumFlux flux>
// std::string Euler<dim, flux>::get_name() {
//     std::stringstream ss;
//     ss << "Euler<" << dim << ", " << "HLLC" << ">"; // TODO make it more gereral
//     return ss.str();
// }

// template<unsigned dim, InviscNumFlux flux> DG_KOKKOS_INLINE_FUNCTION
// void Euler<dim, flux>::vol_fluxes(
//     const EquationInput &input,
//     const Physics::IdealGasParams &params,
//     const rtype *U,
//     const rtype *gU,
//     rtype *F,
//     rtype *gF) {

//     // call the parent class static method
//     EulerBase<dim>::vol_fluxes(input, params, U, gU, F, gF);
// }

// template<unsigned dim, InviscNumFlux flux> DG_KOKKOS_INLINE_FUNCTION
// void Euler<dim, flux>::surf_fluxes(
//     const EquationInput &input,
//     const Physics::IdealGasParams &params,
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F,
//     rtype *gUL,
//     rtype *gUR) {

//     Euler<dim,  flux>::Fchat(params, UL, UR, N, F);
//     Math::HandWritten::cA_to_A<NS>(-1., F);
// }

// template<> DG_KOKKOS_INLINE_FUNCTION
// void Euler<2, InviscNumFlux::HLLC>::Fchat(
//     const Physics::IdealGasParams &params,
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     // common value of gamma
//     const rtype gam = params.gam;

//     // true normal (unity norm)
//     const rtype djac = sqrt(Math::HandWritten::dot<2>(N, N));
//     const rtype djac1 = 1. / djac;
//     const rtype n[2] = {N[0] * djac1, N[1] * djac1};

//     // left state
//     const rtype rhoL = UL[0];
//     const rtype rhoL1 = 1. / rhoL;
//     const rtype unL = (UL[1]*n[0] + UL[2]*n[1]) * rhoL1;
//     const rtype PL = Physics::IdealGas<2>::pressure(params, UL);
//     const rtype aL = sqrt(gam * PL * rhoL1);

//     // right state
//     const rtype rhoR = UR[0];
//     const rtype rhoR1 = 1. / rhoR;
//     const rtype unR = (UR[1]*n[0] + UR[2]*n[1]) * rhoR1;
//     const rtype PR = Physics::IdealGas<2>::pressure(params, UR);
//     const rtype aR = sqrt(gam * PR * rhoR1);

//     // averages
//     const rtype rAvg = 0.5 * (rhoL + rhoR);
//     const rtype aAvg = 0.5 * (aL + aR);

//     // pressure in the star region
//     const rtype pStar = fmax( 0., 0.5 * (PL + PR - (unR-unL)*rAvg*aAvg) );

//     // wave speed estimates using PVRS
//     rtype qL = 1.0;
//     if (pStar / PL > 1.) qL = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PL-1.));
//     rtype qR = 1.0;
//     if (pStar / PR > 1.) qR = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PR-1.));
//     const rtype sL = unL - aL*qL;
//     const rtype sR = unR + aR*qR;
//     const rtype sStar = (PR-PL + rhoL*unL*(sL-unL) - rhoR*unR*(sR-unR)) /
//         (rhoL*(sL-unL) - rhoR*(sR-unR));

//     if (sL >= 0.) {
//         F[0] = djac * rhoL*unL;
//         for (int i=0; i<2; i++) {
//             F[1+i] = djac * (UL[1+i]*unL + PL*n[i]);
//         }
//         F[3] = djac * ((UL[3] + PL) * unL);
//     }
//     else if (sR <= 0.) {
//         F[0] = djac * rhoR*unR;
//         for (int i=0; i<2; i++) {
//             F[1+i] = djac * (UR[1+i]*unR + PR*n[i]);
//         }
//         F[3] = djac * (UR[3] + PR) * unR;
//     }
//     else if ((sL<0.) && (sStar>=0.)) {
//         const rtype c = (sL-unL) / (sL-sStar);
//         const rtype factor1 = sStar - unL;
//         const rtype factor2 = sStar + PL / (rhoL*(sL-unL));

//         F[0] = djac * (rhoL*unL + rhoL*sL*(c-1.));
//         for (int i=0; i<2; i++) {
//             F[1+i] = djac * (
//                 UL[1+i]*unL + PL*n[i] +
//                     sL*(UL[1+i]*(c-1.) + rhoL*c*factor1*n[i]) );
//         }
//         F[3] = djac * (
//             (UL[3] + PL) * unL +
//                 sL * (UL[3]*(c-1.) + rhoL*c*factor1*factor2) );
//     }
//     else if ((sStar<0.) && (sR>0.)) {
//         const rtype c = (sR-unR) / (sR-sStar);
//         const rtype factor1 = sStar - unR;
//         const rtype factor2 = sStar + PR / (rhoR*(sR-unR));

//         F[0] = djac * (rhoR*unR + rhoR*sR*(c-1.));
//         for (int i=0; i<2; i++) {
//             F[1+i] = djac * (
//                 UR[1+i]*unR + PR*n[i] +
//                     sR*(UR[1+i]*(c-1.) + rhoR*c*factor1*n[i]) );
//         }
//         F[3] = djac * (
//             (UR[3] + PR) * unR +
//                 sR * (UR[3]*(c-1.) + rhoR*c*factor1*factor2) );
//     }
//     else {
//         char msg[1024];
//         sprintf(msg, "Error in Euler::HLLC\n");
//         for (unsigned i=0; i<NS; i++) {
//             sprintf(msg+strlen(msg), "UL[%d] = %+.5e\tUR[%d] = %+.5e\n", i, UL[i], i, UR[i]);
//         }
//         sprintf(msg+strlen(msg), "PL    = %+.5e\tPR    = %+.5e\n", PL, PR);
//         printf("%s", msg);
//         assert(false); // temporary way of "throwing a fatal exception"...
//     }
// }

// template<> DG_KOKKOS_INLINE_FUNCTION
// void Euler<3, InviscNumFlux::HLLC>::Fchat(
//     const Physics::IdealGasParams &params,
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     // common value of gamma
//     const rtype gam = params.gam;

//     // true normal (unity norm)
//     const rtype djac = sqrt(Math::HandWritten::dot<3>(N, N));
//     const rtype djac1 = 1. / djac;
//     const rtype n[3] = {N[0] * djac1, N[1] * djac1, N[2] * djac1};


//     // left state
//     const rtype rhoL = UL[0];
//     const rtype rhoL1 = 1. / rhoL;
//     const rtype unL = Math::HandWritten::dot<3>(UL+1, n) * rhoL1;
//     const rtype PL = Physics::IdealGas<3>::pressure(params, UL);
//     const rtype aL = sqrt(gam * PL * rhoL1);

//     // right state
//     const rtype rhoR = UR[0];
//     const rtype rhoR1 = 1. / rhoR;
//     const rtype unR = Math::HandWritten::dot<3>(UR+1, n) * rhoR1;
//     const rtype PR = Physics::IdealGas<3>::pressure(params, UR);
//     const rtype aR = sqrt(gam * PR * rhoR1);

//     // averages
//     const rtype rAvg = 0.5 * (rhoL + rhoR);
//     const rtype aAvg = 0.5 * (aL + aR);

//     // pressure in the star region
//     const rtype pStar = fmax( 0., 0.5 * (PL + PR - (unR-unL)*rAvg*aAvg) );

//     // wave speed estimates using PVRS
//     rtype qL = 1.0;
//     if (pStar / PL > 1.) qL = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PL-1.));
//     rtype qR = 1.0;
//     if (pStar / PR > 1.) qR = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PR-1.));
//     const rtype sL = unL - aL*qL;
//     const rtype sR = unR + aR*qR;
//     const rtype sStar = (PR-PL + rhoL*unL*(sL-unL) - rhoR*unR*(sR-unR)) /
//         (rhoL*(sL-unL) - rhoR*(sR-unR));

//     if (sL >= 0.) {
//         F[0] = djac * rhoL*unL;
//         for (int i=0; i<3; i++) {
//             F[1+i] = djac * (UL[1+i]*unL + PL*n[i]);
//         }
//         F[4] = djac * ((UL[4] + PL) * unL);
//     }
//     else if (sR <= 0.) {
//         F[0] = djac * rhoR*unR;
//         for (int i=0; i<3; i++) {
//             F[1+i] = djac * (UR[1+i]*unR + PR*n[i]);
//         }
//         F[4] = djac * (UR[4] + PR) * unR;
//     }
//     else if ((sL<0.) && (sStar>=0.)) {
//         const rtype c = (sL-unL) / (sL-sStar);
//         const rtype factor1 = sStar - unL;
//         const rtype factor2 = sStar + PL / (rhoL*(sL-unL));

//         F[0] = djac * (rhoL*unL + rhoL*sL*(c-1.));
//         for (int i=0; i<3; i++) {
//             F[1+i] = djac * (
//                 UL[1+i]*unL + PL*n[i] +
//                     sL*(UL[1+i]*(c-1.) + rhoL*c*factor1*n[i]) );
//         }
//         F[4] = djac * (
//             (UL[4] + PL) * unL +
//                 sL * (UL[4]*(c-1.) + rhoL*c*factor1*factor2) );
//     }
//     else if ((sStar<0.) && (sR>0.)) {
//         const rtype c = (sR-unR) / (sR-sStar);
//         const rtype factor1 = sStar - unR;
//         const rtype factor2 = sStar + PR / (rhoR*(sR-unR));

//         F[0] = djac * (rhoR*unR + rhoR*sR*(c-1.));
//         for (int i=0; i<3; i++) {
//             F[1+i] = djac * (
//                 UR[1+i]*unR + PR*n[i] +
//                     sR*(UR[1+i]*(c-1.) + rhoR*c*factor1*n[i]) );
//         }
//         F[4] = djac * (
//             (UR[4] + PR) * unR +
//                 sR * (UR[4]*(c-1.) + rhoR*c*factor1*factor2) );
//     }
//     else {
//         char msg[1024];
//         sprintf(msg, "Error in Euler::HLLC\n");
//         for (unsigned i=0; i<NS; i++) {
//             sprintf(msg+strlen(msg), "UL[%d] = %+.5e\tUR[%d] = %+.5e\n", i, UL[i], i, UR[i]);
//         }
//         sprintf(msg+strlen(msg), "PL    = %+.5e\tPR    = %+.5e\n", PL, PR);
//         printf("%s", msg);
//         assert(false); // temporary way of "throwing a fatal exception"...
//     }
// }

// //--------------------------------------------------------------------------------------------------
// //--------------------------------------------------------------------------------------------------

// template<unsigned dim, InviscNumFlux flux>
// Euler<dim, flux>::Euler(const Physics::IdealGasParams &data) : EulerBase<dim>(data) {}

// template<unsigned dim, InviscNumFlux flux>
// void Euler<dim, flux>::get_surf_fluxes(
//     const EquationInput &input,
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *gUL,
//     const rtype *gUR,
//     const rtype *N,
//     rtype *F,
//     rtype *gFL,
//     rtype *gFR) {

//     rtype Fhat[NS];
//     get_Fchat(UL, UR, N, Fhat);
//     Math::cA_to_B<NS>(-1, Fhat, F);
// }

// template<unsigned dim, InviscNumFlux flux>
// void Euler<dim, flux>::get_weakriemannfullstate_boundary_fluxes(
//     const EquationInput &input, const BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     assert(false);

//     // assert(bdata.has_data);
//     // rtype Ughost[NS];
//     // std::copy_n(bdata.data, NS, Ughost);
//     //
//     // rtype Fhat[NS];
//     // get_Fchat(U, Ughost, N, Fhat);
//     // Math::cA_to_B(NS, -1, Fhat, F);
// }

// template<unsigned dim, InviscNumFlux flux>
// void Euler<dim, flux>::get_weakriemannslipwall_boundary_fluxes(
//     const EquationInput &input, const BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     assert(false);

//     // rtype Ughost[NS];
//     // std::copy_n(U, NS, Ughost);
//     //
//     // // get true normalized normal
//     // rtype n[dim];
//     // rtype N_norm = sqrt(Math::dot<dim>(N, N));
//     // for (unsigned idim=0; idim<dim; idim++) {
//     //     n[idim] = N[idim] / N_norm;
//     // }
//     //
//     // // mirror momentum
//     // rtype rhoun = Math::dot<dim>(U+1, n);
//     // for (unsigned idim=0; idim<dim; idim++) {
//     //     Ughost[1+idim] -= 2. * rhoun * n[idim];
//     // }
//     //
//     // rtype Fhat[NS];
//     // get_Fchat(U, Ughost, N, Fhat);
//     // Math::cA_to_B(NS, -1, Fhat, F);
// }

// template<unsigned dim, InviscNumFlux flux>
// void Euler<dim, flux>::get_weakriemannpressureoutlet_boundary_fluxes(
//     const EquationInput &input, const BoundaryData &bdata,
//     const rtype *U, const rtype *gU, const rtype *N, rtype *F, rtype *gF) {

//     assert(false);

//     // assert(bdata.has_data);
//     // const rtype Pb = bdata.data[0];
//     // const rtype gam = this->phys.get_gamma();
//     //
//     // // extrapolate density and momentum
//     // rtype Ughost[NS];
//     // std::copy_n(U, dim+1, Ughost);
//     // Ughost[dim+1] = Pb/(gam-1) + 0.5*Math::dot<dim>(U+1,U+1)/U[0];
//     //
//     // rtype Fhat[NS];
//     // get_Fchat(U, Ughost, N, Fhat);
//     // Math::cA_to_B(NS, -1, Fhat, F);
// }

// //--------------------------------------------------------------------------------------------------
// //--------------------------------------------------------------------------------------------------

// template<> inline
// void Euler<2, InviscNumFlux::HLLC>::get_Fchat(
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     const rtype gam = phys.get_gamma();
//     const rtype PL = phys.comp_P(UL);
//     const rtype PR = phys.comp_P(UR);
//     conv_num_flux_HLLC<2>(UL, UR, PL, PR, gam, N, F);
// }

// template<> inline
// void Euler<3, InviscNumFlux::HLLC>::get_Fchat(
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     const rtype gam = phys.get_gamma();
//     const rtype PL = phys.comp_P(UL);
//     const rtype PR = phys.comp_P(UR);
//     conv_num_flux_HLLC<3>(UL, UR, PL, PR, gam, N, F);\
// }

// template<> inline
// void Euler<2, InviscNumFlux::Roe>::get_Fchat(
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     const rtype gam = phys.get_gamma();
//     const rtype PL = phys.comp_P(UL);
//     const rtype PR = phys.comp_P(UR);
//     conv_num_flux_Roe<2>(UL, UR, PL, PR, gam, N, F);
// }

// template<> inline
// void Euler<3, InviscNumFlux::Roe>::get_Fchat(
//     const rtype *UL,
//     const rtype *UR,
//     const rtype *N,
//     rtype *F) {

//     const rtype gam = phys.get_gamma();
//     const rtype PL = phys.comp_P(UL);
//     const rtype PR = phys.comp_P(UR);
//     conv_num_flux_Roe<3>(UL, UR, PL, PR, gam, N, F);
// }

template class Euler<2>;
template class Euler<3>;

} // namespace Physics
