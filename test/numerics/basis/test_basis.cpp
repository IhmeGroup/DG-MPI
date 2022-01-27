#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "numerics/basis/basis.h"

constexpr rtype DOUBLE_TOL = 1E-13;

/*
Test the return for the get_num_basis_coeff function for LegendreSeg
*/
TEST(basis_test_suite, test_get_num_basis_coeff_legendreseg){
	for (int order = 0; order < 10; order++){
		Basis::LegendreSeg basis(order);
		int nb = basis.get_num_basis_coeff(order);

		ASSERT_EQ(nb, order + 1);
	}
}