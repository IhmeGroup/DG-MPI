
#include <cstdlib>
#include "common/defines.h"
#include "gtest/gtest.h"
#include "physics/euler/euler.h"
// #include "equations/navier_stokes.h"
// #include "equations/navier_stokes_multispecies.h"

#include <Kokkos_Core.hpp>

constexpr rtype DOUBLE_TOL = 1E-13;

/*
Test the 2D convective physical flux function for the Euler equations
*/
TEST(PhysicsTestSuite, ConvFluxPhysical2D) {
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = 3.5;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<2> physics;
    physics.set_physical_params();

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
    physics.conv_flux_physical(U, F);
  
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 2; j++){
            EXPECT_NEAR(F(i, j), Fref(i, j), DOUBLE_TOL);
        }
    }
}

/*
Test the 3D convective physical flux function for the Euler equations
*/
TEST(PhysicsTestSuite, ConvFluxPhysical3D) {
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = 3.5;
    const rtype w = -4.5;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<3> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[5]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rho * w;
    U(4) = P / (gam - 1.) + 0.5 * rho * (u * u + v * v + w * w);

    Kokkos::View<rtype[4][3]> Fref("Fref");
    Fref(0, 0) = rho * u;
    Fref(1, 0) = rho * u * u + P;
    Fref(2, 0) = rho * u * v;
    Fref(3, 0) = rho * u * w;
    Fref(4, 0) = (rhoE + P) * u;
    Fref(0, 1) = rho * v;
    Fref(1, 1) = rho * u * v;
    Fref(2, 1) = rho * v * v + P;
    Fref(3, 1) = rho * v * w;
    Fref(4, 1) = (rhoE + P) * v;
    Fref(0, 2) = rho * w;
    Fref(1, 2) = rho * u * w;
    Fref(2, 2) = rho * v * w;
    Fref(3, 2) = rho * w * w + P;
    Fref(4, 2) = (rhoE + P) * w;

    Kokkos::View<rtype[4][3]> F("F");
    physics.conv_flux_physical(U, F);
  
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 3; j++){
            EXPECT_NEAR(F(i, j), Fref(i, j), DOUBLE_TOL);
        }
    }
}

/*
Test the 2D projected physical flux function for the Euler equations

This function sets the velocity vector orthogonal to the normal and
checks it against the analytical solution.
*/
TEST(PhysicsTestSuite, ConvFluxPhysicalProjected2DOrthogonalTest1) {
    const rtype P = 1.0;
    const rtype rho = 1.0;
    const rtype u = 1.0;
    const rtype v = 0.0;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<2> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    Kokkos::View<rtype[2]> normals("normals");
    normals(0) = 0.0;
    normals(1) = 1.0;

    // reference solution hand calculated
    Kokkos::View<rtype[4]> Fref("Fref");
    Fref(0) = 0.0;
    Fref(1) = 0.0;
    Fref(2) = 1.0;
    Fref(3) = 0.0;

    Kokkos::View<rtype[4]> F("F");
    physics.get_conv_flux_projected(U, normals, F);

    for (unsigned i = 0; i < 4; i++) {
        EXPECT_NEAR(F(i), Fref(i), DOUBLE_TOL);
    }
}

/*
Test the 2D projected physical flux function for the Euler equations

This function sets the velocity vector orthogonal to the normal and
checks it against the analytical solution.
*/
TEST(PhysicsTestSuite, ConvFluxPhysicalProjected2DOrthogonalTest2) {
    const rtype P = 1.0;
    const rtype rho = 1.0;
    const rtype u = 0.0;
    const rtype v = 1.0;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<2> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    Kokkos::View<rtype[2]> normals("normals");
    normals(0) = 1.0;
    normals(1) = 0.0;

    // reference solution hand calculated
    Kokkos::View<rtype[4]> Fref("Fref");
    Fref(0) = 0.0;
    Fref(1) = 1.0;
    Fref(2) = 0.0;
    Fref(3) = 0.0;

    Kokkos::View<rtype[4]> F("F");
    physics.get_conv_flux_projected(U, normals, F);

    for (unsigned i = 0; i < 4; i++) {
        EXPECT_NEAR(F(i), Fref(i), DOUBLE_TOL);
    }
}

/*
Test the 3D projected physical flux function for the Euler equations

This function sets the velocity vector orthogonal to the normal and
checks it against the analytical solution.
*/
TEST(PhysicsTestSuite, ConvFluxPhysicalProjected3DOrthogonalTest1) {
    const rtype P = 1.0;
    const rtype rho = 1.0;
    const rtype u = 1.0;
    const rtype v = 0.0;
    const rtype w = 0.0;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v + w * w);

    Physics::Euler<3> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[5]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rho * w;
    U(4) = rhoE;

    Kokkos::View<rtype[3]> normals("normals");
    normals(0) = 0.0;
    normals(1) = 0.5;
    normals(2) = 0.5;

    // reference solution hand calculated
    Kokkos::View<rtype[5]> Fref("Fref");
    Fref(0) = 0.0;
    Fref(1) = 0.0;
    Fref(2) = 0.5;
    Fref(3) = 0.5;
    Fref(4) = 0.0;

    Kokkos::View<rtype[5]> F("F");
    physics.get_conv_flux_projected(U, normals, F);

    for (unsigned i = 0; i < 5; i++) {
        EXPECT_NEAR(F(i), Fref(i), DOUBLE_TOL);
    }
}

/*
Test the 2D projected physical flux function for the Euler equations

This function sets the velocity vector parralel to the normal and
checks it against the analytical solution.
*/
TEST(PhysicsTestSuite, ConvFluxPhysicalProjected2DParallelTest1) {
    const rtype P = 1.0;
    const rtype rho = 1.0;
    const rtype u = 1.0;
    const rtype v = 0.0;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<2> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    Kokkos::View<rtype[2]> normals("normals");
    normals(0) = 1.0;
    normals(1) = 0.0;

    // reference solution hand calculated
    Kokkos::View<rtype[4]> Fref("Fref");
    Fref(0) = 1.0;
    Fref(1) = 2.0;
    Fref(2) = 0.0;
    Fref(3) = 4.0;

    Kokkos::View<rtype[4]> F("F");
    physics.get_conv_flux_projected(U, normals, F);

    for (unsigned i = 0; i < 4; i++) {
        EXPECT_NEAR(F(i), Fref(i), DOUBLE_TOL);
    }
}

/*
Test the 2D projected physical flux function for the Euler equations

This function sets the velocity vector parralel to the normal and
checks it against the analytical solution.
*/
TEST(PhysicsTestSuite, ConvFluxPhysicalProjected2DParallelTest2) {
    const rtype P = 1.0;
    const rtype rho = 1.0;
    const rtype u = 0.0;
    const rtype v = 1.0;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

    Physics::Euler<2> physics;
    physics.set_physical_params();

    Kokkos::initialize();

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    Kokkos::View<rtype[2]> normals("normals");
    normals(0) = 0.0;
    normals(1) = 1.0;

    // reference solution hand calculated
    Kokkos::View<rtype[4]> Fref("Fref");
    Fref(0) = 1.0;
    Fref(1) = 0.0;
    Fref(2) = 2.0;
    Fref(3) = 4.0;

    Kokkos::View<rtype[4]> F("F");
    physics.get_conv_flux_projected(U, normals, F);

    for (unsigned i = 0; i < 4; i++) {
        EXPECT_NEAR(F(i), Fref(i), DOUBLE_TOL);
    }
}


TEST(PhysicsTestSuite, SetConstantPropertiesEuler) {
    const rtype gamref = 1.6;
    const rtype Rref = 300.0;

    Physics::Euler<2> physics;

    physics.set_physical_params(Rref, gamref);

    EXPECT_NEAR(gamref, physics.gamma, DOUBLE_TOL);
    EXPECT_NEAR(Rref, physics.R, DOUBLE_TOL);
}

TEST(PhysicsTestSuite, SetDefaultConstantPropertiesEuler) {
    const rtype gamref = 1.4;
    const rtype Rref = 287.0;

    Physics::Euler<2> physics;

    physics.set_physical_params();

    EXPECT_NEAR(gamref, physics.gamma, DOUBLE_TOL);
    EXPECT_NEAR(Rref, physics.R, DOUBLE_TOL);
}

TEST(PhysicsTestSuite, GetPressure2D) {
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = 3.5;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);


    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    Physics::Euler<2> physics;
    physics.set_physical_params();

    rtype Pcalc = physics.get_pressure(U);

    EXPECT_NEAR(P, Pcalc, DOUBLE_TOL);        
}

TEST(PhysicsTestSuite, GetPressure3D) {
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = 3.5;
    const rtype w = -4.5;
    const rtype gam = 1.4;
    const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v + w * w);

    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rho * w;
    U(4) = rhoE;

    Physics::Euler<3> physics;
    physics.set_physical_params();

    rtype Pcalc = physics.get_pressure(U);

    EXPECT_NEAR(P, Pcalc, DOUBLE_TOL);        
}
