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

template<unsigned dim> KOKKOS_INLINE_FUNCTION
rtype get_maxwavespeed(const rtype& gamma, const rtype* U) {
    // unpack
    rtype mom[dim];

    // slice the state
    const int start = 1;
    for (unsigned i = 0; i < dim; i++){
        mom[i] = U[start + i];
    }
    // specific volume
    auto rho1 = 1./U[0];
    const rtype sqrtrKE = sqrt(Math::dot<dim>(mom, mom)) * rho1;
    
    return sqrtrKE + sqrt(gamma * get_pressure<dim>(gamma, U) * rho1); 
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

template<typename ViewTypeX, typename ViewTypeUq> KOKKOS_INLINE_FUNCTION
void set_state_isentropic_vortex(const Physics::Physics<3>* physics, ViewTypeX x, const rtype t,
    ViewTypeUq Uq){
    // quantities used to make it dimensionless
    rtype rho0 = 1.;
    rtype sos0 = 1.;
    rtype L = 1.;

    rtype mom[3];

    // parameters
    rtype Mach = 1.0;
    rtype radius = L / 10.;
    rtype sigma = 1;

    // other quantities
    rtype gamma = physics->get_gamma();
    rtype beta = Mach * 5. * sqrt(2.) / (4.*M_PI) * exp(0.5);
    rtype u0 = Mach * sos0;
    // NOTE: For now we assume that we compare this to the exact solution after 1 period
    rtype x1 = x(0) - 0.5;
    rtype y1 = x(2) - 0.5;
    rtype f = -0.5/(sigma*sigma) * (x1*x1 + y1*y1) / (radius*radius);
    rtype Omega = beta * exp(f);

    // perturbations
    rtype du = -y1/radius*Omega;
    rtype dv =  x1/radius*Omega;
    rtype dT = -(gamma-1.)/2. * Omega*Omega;

    // initialize
    rtype p =1./gamma * pow(1+dT, gamma/(gamma-1.));
    Uq(0) = pow(rho0 + dT, 1./(gamma-1.));
    Uq(1) = Uq(0) * (u0 + du) ;
    Uq(2) = 0.;
    Uq(3) = Uq(0) * dv ;

    // slice the state 
    const int start = 1;
    for (unsigned i = 0; i < 3; i++){
        mom[i] = Uq(start + i);
    }
    const rtype rKE = Math::dot<3>(mom, mom) / Uq(0);
    Uq(4) = p / (gamma - 1.0) + 0.5 * rKE;
}
} // end namespace EulerFcnType

namespace EulerConvNumFluxType {

template<> KOKKOS_INLINE_FUNCTION
void compute_flux_hllc(const Physics::Physics<2>& physics,
    const rtype* UL, const rtype* UR, const rtype* N, 
    rtype* F, rtype* gUL, rtype* gUR) {

    static constexpr int NS = physics.NUM_STATE_VARS;

    const rtype gam = physics.gamma;

    // true normal (unity norm)
    const rtype djac = sqrt(Math::dot<2>(N, N));
    const rtype djac1 = 1. / djac;
    const rtype n[2] = {N[0] * djac1, N[1] * djac1};

    // left state
    const rtype rhoL = UL[0];
    const rtype rhoL1 = 1. / rhoL;
    const rtype unL = (UL[1]*n[0] + UL[2]*n[1]) * rhoL1;
    const rtype PL = EulerFcnType::get_pressure<2>(gam, UL);
    const rtype aL = sqrt(gam * PL * rhoL1);

    // right state
    const rtype rhoR = UR[0];
    const rtype rhoR1 = 1. / rhoR;
    const rtype unR = (UR[1]*n[0] + UR[2]*n[1]) * rhoR1;
    const rtype PR = EulerFcnType::get_pressure<2>(gam, UR);
    const rtype aR = sqrt(gam * PR * rhoR1);

    // averages
    const rtype rAvg = 0.5 * (rhoL + rhoR);
    const rtype aAvg = 0.5 * (aL + aR);

    // pressure in the star region
    const rtype pStar = fmax( 0., 0.5 * (PL + PR - (unR-unL)*rAvg*aAvg) );

    // wave speed estimates using PVRS
    rtype qL = 1.0;
    if (pStar / PL > 1.) qL = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PL-1.));
    rtype qR = 1.0;
    if (pStar / PR > 1.) qR = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PR-1.));
    const rtype sL = unL - aL*qL;
    const rtype sR = unR + aR*qR;
    const rtype sStar = (PR-PL + rhoL*unL*(sL-unL) - rhoR*unR*(sR-unR)) /
        (rhoL*(sL-unL) - rhoR*(sR-unR));

    if (sL >= 0.) {
        F[0] = djac * rhoL*unL;
        for (int i=0; i<2; i++) {
            F[1+i] = djac * (UL[1+i]*unL + PL*n[i]);
        }
        F[3] = djac * ((UL[3] + PL) * unL);
    }
    else if (sR <= 0.) {
        F[0] = djac * rhoR*unR;
        for (int i=0; i<2; i++) {
            F[1+i] = djac * (UR[1+i]*unR + PR*n[i]);
        }
        F[3] = djac * (UR[3] + PR) * unR;
    }
    else if ((sL<0.) && (sStar>=0.)) {
        const rtype c = (sL-unL) / (sL-sStar);
        const rtype factor1 = sStar - unL;
        const rtype factor2 = sStar + PL / (rhoL*(sL-unL));

        F[0] = djac * (rhoL*unL + rhoL*sL*(c-1.));
        for (int i=0; i<2; i++) {
            F[1+i] = djac * (
                UL[1+i]*unL + PL*n[i] +
                    sL*(UL[1+i]*(c-1.) + rhoL*c*factor1*n[i]) );
        }
        F[3] = djac * (
            (UL[3] + PL) * unL +
                sL * (UL[3]*(c-1.) + rhoL*c*factor1*factor2) );
    }
    else if ((sStar<0.) && (sR>0.)) {
        const rtype c = (sR-unR) / (sR-sStar);
        const rtype factor1 = sStar - unR;
        const rtype factor2 = sStar + PR / (rhoR*(sR-unR));

        F[0] = djac * (rhoR*unR + rhoR*sR*(c-1.));
        for (int i=0; i<2; i++) {
            F[1+i] = djac * (
                UR[1+i]*unR + PR*n[i] +
                    sR*(UR[1+i]*(c-1.) + rhoR*c*factor1*n[i]) );
        }
        F[3] = djac * (
            (UR[3] + PR) * unR +
                sR * (UR[3]*(c-1.) + rhoR*c*factor1*factor2) );
    }
    else {
        char msg[1024];
        sprintf(msg, "Error in Euler::HLLC\n");
        for (unsigned i=0; i<NS; i++) {
            sprintf(msg+strlen(msg), "UL[%d] = %+.5e\tUR[%d] = %+.5e\n", i, UL[i], i, UR[i]);
        }
        sprintf(msg+strlen(msg), "PL    = %+.5e\tPR    = %+.5e\n", PL, PR);
        printf("%s", msg);
        assert(false); // temporary way of "throwing a fatal exception"...
    }

} // end compute_flux_hllc (2d specialization)

template<> KOKKOS_INLINE_FUNCTION
void compute_flux_hllc(const Physics::Physics<3>& physics,
    const rtype* UL, const rtype* UR, const rtype* N, 
    rtype* F, rtype* gUL, rtype* gUR) {
    
    static constexpr int NS = physics.NUM_STATE_VARS;

    // common value of gamma
    const rtype gam = physics.gamma;

    // true normal (unity norm)
    const rtype djac = sqrt(Math::dot<3>(N, N));
    const rtype djac1 = 1. / djac;
    const rtype n[3] = {N[0] * djac1, N[1] * djac1, N[2] * djac1};


    // left state
    const rtype rhoL = UL[0];
    const rtype rhoL1 = 1. / rhoL;
    const rtype unL = Math::dot<3>(UL+1, n) * rhoL1;
    const rtype PL = EulerFcnType::get_pressure<3>(gam, UL);
    const rtype aL = sqrt(gam * PL * rhoL1);

    // right state
    const rtype rhoR = UR[0];
    const rtype rhoR1 = 1. / rhoR;
    const rtype unR = Math::dot<3>(UR+1, n) * rhoR1;
    const rtype PR = EulerFcnType::get_pressure<3>(gam, UR);
    const rtype aR = sqrt(gam * PR * rhoR1);

    // averages
    const rtype rAvg = 0.5 * (rhoL + rhoR);
    const rtype aAvg = 0.5 * (aL + aR);

    // pressure in the star region
    const rtype pStar = fmax( 0., 0.5 * (PL + PR - (unR-unL)*rAvg*aAvg) );

    // wave speed estimates using PVRS
    rtype qL = 1.0;
    if (pStar / PL > 1.) qL = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PL-1.));
    rtype qR = 1.0;
    if (pStar / PR > 1.) qR = sqrt(1. + (gam+1.)/(2.*gam) * (pStar/PR-1.));
    const rtype sL = unL - aL*qL;
    const rtype sR = unR + aR*qR;
    const rtype sStar = (PR-PL + rhoL*unL*(sL-unL) - rhoR*unR*(sR-unR)) /
        (rhoL*(sL-unL) - rhoR*(sR-unR));

    if (sL >= 0.) {
        F[0] = djac * rhoL*unL;
        for (int i=0; i<3; i++) {
            F[1+i] = djac * (UL[1+i]*unL + PL*n[i]);
        }
        F[4] = djac * ((UL[4] + PL) * unL);
    }
    else if (sR <= 0.) {
        F[0] = djac * rhoR*unR;
        for (int i=0; i<3; i++) {
            F[1+i] = djac * (UR[1+i]*unR + PR*n[i]);
        }
        F[4] = djac * (UR[4] + PR) * unR;
    }
    else if ((sL<0.) && (sStar>=0.)) {
        const rtype c = (sL-unL) / (sL-sStar);
        const rtype factor1 = sStar - unL;
        const rtype factor2 = sStar + PL / (rhoL*(sL-unL));

        F[0] = djac * (rhoL*unL + rhoL*sL*(c-1.));
        for (int i=0; i<3; i++) {
            F[1+i] = djac * (
                UL[1+i]*unL + PL*n[i] +
                    sL*(UL[1+i]*(c-1.) + rhoL*c*factor1*n[i]) );
        }
        F[4] = djac * (
            (UL[4] + PL) * unL +
                sL * (UL[4]*(c-1.) + rhoL*c*factor1*factor2) );
    }
    else if ((sStar<0.) && (sR>0.)) {
        const rtype c = (sR-unR) / (sR-sStar);
        const rtype factor1 = sStar - unR;
        const rtype factor2 = sStar + PR / (rhoR*(sR-unR));

        F[0] = djac * (rhoR*unR + rhoR*sR*(c-1.));
        for (int i=0; i<3; i++) {
            F[1+i] = djac * (
                UR[1+i]*unR + PR*n[i] +
                    sR*(UR[1+i]*(c-1.) + rhoR*c*factor1*n[i]) );
        }
        F[4] = djac * (
            (UR[4] + PR) * unR +
                sR * (UR[4]*(c-1.) + rhoR*c*factor1*factor2) );
    }
    else {
        char msg[1024];
        sprintf(msg, "Error in Euler::HLLC\n");
        for (unsigned i=0; i<NS; i++) {
            sprintf(msg+strlen(msg), "UL[%d] = %+.5e\tUR[%d] = %+.5e\n", i, UL[i], i, UR[i]);
        }
        sprintf(msg+strlen(msg), "PL    = %+.5e\tPR    = %+.5e\n", PL, PR);
        printf("%s", msg);
        assert(false); // temporary way of "throwing a fatal exception"...
    }
} // end compute_flux_hllc (3d specialization)

}