#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "common/enums.h"
#include "common/defines.h"
#include "math/linear_algebra.h"
#include <Kokkos_Core.hpp>

constexpr rtype DOUBLE_TOL = 1E-13;
constexpr rtype SINGLE_TOL = 1E-5;



TEST(math_suit, test_batched_math_inv_2x2){

const double alpha(1), beta(0);

int N = 1;
int m = 2;

view_type_3D AA("AA", N, m, m);
view_type_3D BB("BB", N, m, m);
view_type_3D CC("CC", N, m, m);

host_view_type_3D h_AA = Kokkos::create_mirror_view(AA);
host_view_type_3D h_BB = Kokkos::create_mirror_view(BB);
host_view_type_3D h_CC = Kokkos::create_mirror_view(CC);

h_AA(0, 0, 0) = 4.; h_AA(0, 1, 0) = 6.;
h_AA(0, 0, 1) = 3.; h_AA(0, 1, 1) = 3.;

h_BB(0, 0, 0) = 1.; h_BB(0, 0, 1) = 0.;
h_BB(0, 1, 0) = 0.; h_BB(0, 1, 1) = 1.;

h_CC(0, 0, 0) = -0.5; h_CC(0, 1, 0) = 1.;
h_CC(0, 0, 1) = 0.5; h_CC(0, 1, 1) = -2./3.;

Kokkos::deep_copy(AA, h_AA);
Kokkos::deep_copy(BB, h_BB);


Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int &idx) {
	auto A = Kokkos::subview(AA, idx, Kokkos::ALL(), Kokkos::ALL());
	auto B = Kokkos::subview(BB, idx, Kokkos::ALL(), Kokkos::ALL());
	Math::invA(A, B);
});

Kokkos::deep_copy(h_BB, BB);

for (int i = 0; i < 2; i++){
	for (int j = 0; j < 2; j++){
		EXPECT_NEAR(h_BB(0, i, j), h_CC(0, i, j), DOUBLE_TOL);
	}
}
}


TEST(math_suit, test_batched_math_inv_3x3){

const double alpha(1), beta(0);

int N = 1;
int m = 3;

view_type_3D AA("AA", N, m, m);
view_type_3D BB("BB", N, m, m);
view_type_3D CC("CC", N, m, m);

host_view_type_3D h_AA = Kokkos::create_mirror_view(AA);
host_view_type_3D h_BB = Kokkos::create_mirror_view(BB);
host_view_type_3D h_CC = Kokkos::create_mirror_view(CC);

h_AA(0, 0, 0) = 1.; h_AA(0, 1, 0) = 2.; h_AA(0, 2, 0) = 3.;
h_AA(0, 0, 1) = 4.; h_AA(0, 1, 1) = 2.; h_AA(0, 2, 1) = 3.;
h_AA(0, 0, 2) = 7.; h_AA(0, 1, 2) = 8.; h_AA(0, 2, 2) = 9.;


h_BB(0, 0, 0) = 1.; h_BB(0, 1, 0) = 0.; h_BB(0, 2, 0) = 0.;
h_BB(0, 0, 1) = 0.; h_BB(0, 1, 1) = 1.; h_BB(0, 2, 1) = 0.;
h_BB(0, 0, 2) = 0.; h_BB(0, 1, 2) = 0.; h_BB(0, 2, 2) = 1.;

h_CC(0, 0, 0) = -1./3.; h_CC(0, 1, 0) = 1./3.; h_CC(0, 2, 0) = 0.;
h_CC(0, 0, 1) = -5./6.; h_CC(0, 1, 1) = -2./3.; h_CC(0, 2, 1) = 0.5;
h_CC(0, 0, 2) = 1.; h_CC(0, 1, 2) = 1./3.; h_CC(0, 2, 2) = -1./3.;

Kokkos::deep_copy(AA, h_AA);
Kokkos::deep_copy(BB, h_BB);

Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int &idx) {
	auto A = Kokkos::subview(AA, idx, Kokkos::ALL(), Kokkos::ALL());
	auto B = Kokkos::subview(BB, idx, Kokkos::ALL(), Kokkos::ALL());
	Math::invA(A, B);

});

Kokkos::deep_copy(h_BB, BB);

for (int i = 0; i < 2; i++){
	for (int j = 0; j < 2; j++){
		EXPECT_NEAR(h_BB(0, i, j), h_CC(0, i, j), DOUBLE_TOL);
	}
}

}

