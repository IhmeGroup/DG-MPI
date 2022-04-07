// // #include "physics/euler/functions.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <cmath>

namespace EulerFcnType {


template<unsigned dim> KOKKOS_INLINE_FUNCTION
rtype get_pressure(const rtype& gamma, const rtype* U) {
    // unpack
    rtype mom[dim];

    // slice the state 
    const int start = 1;
    for (unsigned i = 0; i < dim; i++){
        mom[i] = U[start + i];
    }

    const rtype rKE = Math::dot<dim>(mom, mom) / U[0];
    return (gamma - 1.0) * (U[dim + 1] - 0.5 * rKE);
}

template<> KOKKOS_INLINE_FUNCTION
void conv_flux_interior<2>(const rtype& gamma, const rtype* U, rtype* Fdir){

    const rtype r1 = 1. / U[0];
    const rtype ru = U[1];
    const rtype rv = U[2];
    const rtype rE = U[3];
    const rtype P = get_pressure<2>(gamma, U);

    Fdir[0] = ru;
    Fdir[1] = ru*ru*r1 + P;
    Fdir[2] = ru*rv*r1;
    Fdir[3] = (rE + P)*ru*r1;

    Fdir[4] = rv;
    Fdir[5] = ru*rv*r1;
    Fdir[6] = rv*rv*r1 + P;
    Fdir[7] = (rE + P)*rv*r1;

}

template<> KOKKOS_INLINE_FUNCTION
void conv_flux_interior<3>(const rtype& gamma, const rtype* U, rtype* Fdir){

    const rtype r1 = 1. / U[0];
    const rtype ru = U[1];
    const rtype rv = U[2];
    const rtype rw = U[3];
    const rtype rE = U[4];
    const rtype P = get_pressure<3>(gamma, U);

    Fdir[0] = ru;
    Fdir[1] = ru*ru*r1 + P;
    Fdir[2] = ru*rv*r1;
    Fdir[3] = ru*rw*r1;
    Fdir[4] = (rE + P)*ru*r1;

    Fdir[5] = rv;
    Fdir[6] = rv*ru*r1;
    Fdir[7] = rv*rv*r1 + P;
    Fdir[8] = rv*rw*r1;
    Fdir[9] = (rE + P)*rv*r1;

    Fdir[10] = rw;
    Fdir[11] = rw*ru*r1;
    Fdir[12] = rw*rv*r1;
    Fdir[13] = rw*rw*r1 + P;
    Fdir[14] = (rE + P)*rw*r1;
}


template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_isentropic_vortex(const Physics::Physics<2>* physics, ViewTypeX x, const rtype t,
    ViewTypeUq Uq){

    const rtype rhob = physics->IC_data[0];
    const rtype ub = physics->IC_data[1];
    const rtype vb = physics->IC_data[2];
    const rtype pb = physics->IC_data[3];
    const rtype vs = physics->IC_data[4];

    const rtype gamma = physics->get_gamma();
    const rtype Rg = physics->get_gasconstant();

    assert(Rg == 1.0);

    // Base temperature
    rtype Tb = pb / (rhob * Rg);

    // Entroy 
    rtype s = pb / (pow(rhob, gamma));

    // Track center of vortex
    rtype xr = x[0] - ub * t;
    rtype yr = x[1] - vb * t;

    rtype r = sqrt(pow(xr, 2) + pow(yr, 2));

    // Perturbations
    rtype dU = vs / (2.0 * M_PI) * exp(0.5 * (1.0 - pow(r, 2)));
    rtype du = dU * -1.0 * yr;
    rtype dv = dU * xr;

    rtype dT = -1.0 * (gamma - 1.0) * pow(vs, 2) / (8.0 * gamma * pow(M_PI, 2))
        * exp(1.0 - pow(r, 2));

    rtype u = ub + du;
    rtype v = vb + dv;
    rtype T = Tb + dT;
    
    // Convert to conservative variables
    rtype rho = pow(T/s, 1.0 / (gamma - 1.0));
    rtype rhou = rho * u;
    rtype rhov = rho * v;
    rtype rhoE = rho * Rg / (gamma - 1.0) * T + 0.5 * 
        (rhou * rhou + rhov * rhov) / rho;

    Uq(0) = rho;
    Uq(1) = rhou;
    Uq(2) = rhov;
    Uq(3) = rhoE;

}

// // KOKKOS_INLINE_FUNCTION
// // void set_state_uniform_2D(const Physics::Physics& physics, scratch_view_1D_rtype x, const rtype t,
// //     scratch_view_1D_rtype Uq){

// //     // int NS = physics.get_NS();
// //     // printf("Why memory bad ...?\n");
// //     // printf("NS = %i\n", physics.NUM_STATE_VARS);
// //     // for (int is = 0; is < physics.NUM_STATE_VARS; is++){
// //     //     Uq(is) = 1.0;
// //     // }
// // }

// // KOKKOS_INLINE_FUNCTION
// inline
// void set_state_uniform_2D(){
//     printf("we called this\n");

//     // int NS = physics.get_NS();
//     // printf("Why memory bad ...?\n");
//     // printf("NS = %i\n", physics.NUM_STATE_VARS);
//     // for (int is = 0; is < physics.NUM_STATE_VARS; is++){
//     //     Uq(is) = 1.0;
//     // }
// }


} // end namespace EulerFcnType