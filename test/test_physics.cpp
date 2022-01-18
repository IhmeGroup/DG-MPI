// //
// // Created by kihiro on 7/29/20.
// //

// #include <cstdlib>
// #include "common/defines.h"
// #include "gtest/gtest.h"

// // #include "my_eigen.h"
// // #include "equations/euler.h"
// // #include "equations/navier_stokes.h"
// // #include "equations/navier_stokes_multispecies.h"

// using namespace std;
// // using namespace Physics;
// // using namespace Equation;
// // using VectorType = Eigen::Matrix<rtype, Eigen::Dynamic, 1>;

// constexpr rtype DOUBLE_TOL = 1E-13;

// TEST(PhysicsTestSuite, DummyTest) {
//     cout << "This is a test" << endl;
//     // const rtype P = 101325.;
//     // const rtype rho = 1.1;
//     // const rtype u = 2.5;
//     // const rtype v = 3.5;
//     // const rtype gam = 1.4;
//     // const rtype rhoE = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

//     // rtype U[4];
//     // U[0] = rho;
//     // U[1] = rho * u;
//     // U[2] = rho * v;
//     // U[3] = P / (gam - 1.) + 0.5 * rho * (u * u + v * v);

//     // rtype Fref[8];
//     // Fref[0] = rho * u;
//     // Fref[1] = rho * u * u + P;
//     // Fref[2] = rho * u * v;
//     // Fref[3] = (rhoE + P) * u;
//     // Fref[4] = rho * v;
//     // Fref[5] = rho * u * v;
//     // Fref[6] = rho * v * v + P;
//     // Fref[7] = (rhoE + P) * v;

//     // rtype F[8];
//     // EulerBase<2>::conv_flux_physical(4, U, P, F);

//     // for (unsigned i = 0; i < 8; i++) {
//     //     EXPECT_NEAR(F[i], Fref[i], DOUBLE_TOL);
//     // }
// }