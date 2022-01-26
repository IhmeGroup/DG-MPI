
#include <cstdlib>
#include <math.h>
#include "common/defines.h"
#include "gtest/gtest.h"
#include "physics/euler/euler.h"
#include "physics/base/functions.h"
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

/* 
Test max wave speed calc for zero velocity field
*/
TEST(PhysicsTestSuite, GetMaxWaveSpeed3DZeroVelocity) {
    const rtype P = 1.;
    const rtype rho = 1.;
    const rtype u = 0.;
    const rtype v = 0.;
    const rtype w = 0.;
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
    rtype acalc = physics.get_maxwavespeed(U);

    EXPECT_NEAR(sqrt(1.4), acalc, DOUBLE_TOL);
}

/* 
Test max wave speed calc
*/
TEST(PhysicsTestSuite, GetMaxWaveSpeed3D) {
    const rtype P = 1.;
    const rtype rho = 1.;
    const rtype u = 1.;
    const rtype v = 2.;
    const rtype w = 2.;
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
    rtype acalc = physics.get_maxwavespeed(U);

    EXPECT_NEAR(3.0 + sqrt(1.4), acalc, DOUBLE_TOL);
}
/*
Test the selection of additional variables via input strings
*/
TEST(PhysicsTestSuite, GetPhysicalVariableEnumCheckTest1) {

    Physics::Euler<2> physics;
    Physics::Euler<2>::PhysicsVariables var = 
        physics.get_physical_variable("MaxWaveSpeed");

    if (var == Physics::Euler<2>::PhysicsVariables::MaxWaveSpeed){
        EXPECT_NEAR(1.0, 1.0, DOUBLE_TOL); // Force pass
    }else{
        EXPECT_NEAR(1.0, 0.0, DOUBLE_TOL); // Force fail
    }
}

/*
Test the selection of additional variables via input strings
*/
TEST(PhysicsTestSuite, GetPhysicalVariableEnumCheckTest2) {

    Physics::Euler<2> physics;
    Physics::Euler<2>::PhysicsVariables var = 
        physics.get_physical_variable("Density");

    if (var == Physics::Euler<2>::PhysicsVariables::Density){
        EXPECT_NEAR(1.0, 1.0, DOUBLE_TOL); // Force pass
    }else{
        EXPECT_NEAR(0.0, 1.0, DOUBLE_TOL); // Force fail
    }
}


/*
Test that ensures that the 2D numerical flux functions are consistent,
i.e. that F_numerical(U, U, normals) = F(U) dot normals
*/
TEST(PhysicsTestSuite, test_numerical_flux_2D_consistency) {
    
    // instantiate physics
    Physics::Euler<2> physics;
    physics.set_physical_params();

    // instantiate flux function -> evenutally this should be done via 
    // enums.
    BaseConvNumFluxType::LaxFriedrichs flux;

    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = -3.5;
    const rtype rhoE = P / (physics.gamma - 1.) + 0.5 * rho * (u * u + v * v);
    
    Kokkos::View<rtype[4]> U("U");
    U(0) = rho;
    U(1) = rho * u;
    U(2) = rho * v;
    U(3) = rhoE;

    // Compute numerical flux
    Kokkos::View<rtype*> normals("normals", 2);
    Kokkos::View<rtype*> Fnum("Fnum", 4);
    Kokkos::View<rtype*> F_expected("F_expected", 4);

    normals(0) = -0.4;
    normals(1) = 0.7;
    flux.compute_flux(physics, U, U, normals, Fnum);

    physics.get_conv_flux_projected(U, normals, F_expected);


    for (unsigned i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(Fnum(i), F_expected(i));
    }
}

/*
Test that ensures that the 2D numerical flux functions are conservative,
i.e. that F_numerical(UL, UR, normals) = -F_numerical(UR, UL, -normals)
*/
TEST(PhysicsTestSuite, test_numerical_flux_2D_conservation) {
    
    // instantiate physics
    Physics::Euler<2> physics;
    physics.set_physical_params();

    // instantiate flux function -> evenutally this should be done via 
    // enums.
    BaseConvNumFluxType::LaxFriedrichs flux;

    // Set the left state
    const rtype P = 101325.;
    const rtype rho = 1.1;
    const rtype u = 2.5;
    const rtype v = -3.5;
    const rtype rhoE = P / (physics.gamma - 1.) + 0.5 * rho * (u * u + v * v);
    
    Kokkos::View<rtype[4]> UqL("U");
    UqL(0) = rho;
    UqL(1) = rho * u;
    UqL(2) = rho * v;
    UqL(3) = rhoE;

    // Set the right state
    const rtype PR = 101325. * 2.;
    const rtype rhoR = 0.7;
    const rtype uR = -3.5;
    const rtype vR = -6.0;
    const rtype rhoER = PR / (physics.gamma - 1.) + 0.5 * rhoR * (uR * uR + vR * vR);
    
    Kokkos::View<rtype[4]> UqR("U");
    UqR(0) = rhoR;
    UqR(1) = rhoR * uR;
    UqR(2) = rhoR * vR;
    UqR(3) = rhoER;

    // Compute numerical flux
    Kokkos::View<rtype*> normalsL("left normals", 2);
    Kokkos::View<rtype*> normalsR("right normals", 2);
    Kokkos::View<rtype*> Fnum("Fnum", 4);
    Kokkos::View<rtype*> F_expected("F_expected", 4);

    normalsL(0) = -0.4;
    normalsL(1) = 0.7;
    normalsR(0) = 0.4;
    normalsR(1) = -0.7;

    flux.compute_flux(physics, UqL, UqR, normalsL, Fnum);
    flux.compute_flux(physics, UqR, UqL, normalsR, F_expected);


    for (unsigned i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(Fnum(i), -1.*F_expected(i));
    }
}