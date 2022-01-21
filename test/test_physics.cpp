//
// Created by kihiro on 7/29/20.
//

#include <cstdlib>
#include "common/defines.h"
#include "gtest/gtest.h"
#include "physics/euler/euler.h"
// #include "my_eigen.h"
// #include "equations/euler.h"
// #include "equations/navier_stokes.h"
// #include "equations/navier_stokes_multispecies.h"

#include <Kokkos_Core.hpp>


// using VectorType = Eigen::Matrix<rtype, Eigen::Dynamic, 1>;

constexpr rtype DOUBLE_TOL = 1E-13;

TEST(PhysicsTestSuite, ConvFluxPhysical) {
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = 3.5;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Kokkos::initialize();

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Kokkos::View<rtype[4][2]> Fref("Fref");
    Fref(0, 0) = rho * u;
    Fref(1, 0) = rho * u * u + P;
    Fref(2, 0) = rho * u * v;
    Fref(3, 0) = (rhoE + P) * u;
    Fref(0, 1) = rho * v;
    Fref(1, 1) = rho * u * v;
    Fref(2, 1) = rho * v * v + P;
    Fref(3, 1) = (rhoE + P) * v;

    Kokkos::View<rtype[4][2]> F("F");
    Physics::EulerBase<2>::conv_flux_physical(U, P, F);
  
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 2; j++){
            EXPECT_NEAR(F(i, j), Fref(i, j), DOUBLE_TOL);
        }
    }
}