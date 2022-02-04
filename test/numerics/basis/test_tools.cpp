#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "numerics/basis/tools.h"

constexpr rtype DOUBLE_TOL = 1E-13;


/*
Tests that one point returns a zero size 
*/
TEST(basis_tools_test_suite, test_1D_nodes_one_point){
	
	Kokkos::View<rtype*> xnodes("xnodes", 1);
	BasisTools::equidistant_nodes_1D_range(0.0, 10.0, 1, xnodes);
	Kokkos::View<rtype*> expected("expected", 1);
	expected(0) = 0.0; 

	for (int i = 0; i < 2; i++){
		EXPECT_NEAR(expected(i), xnodes(i), DOUBLE_TOL);
	}
}

/*
Tests that two points remain at their locations 
*/
TEST(basis_tools_test_suite, test_1D_nodes_two_points){
	
	Kokkos::View<rtype*> xnodes("xnodes", 1);
	BasisTools::equidistant_nodes_1D_range(0.0, 10.0, 2, xnodes);
	Kokkos::View<rtype*> expected("expected", 2);
	expected(0) = 0.0; 
	expected(1) = 10.0;

	for (int i = 0; i < 2; i++){
		EXPECT_NEAR(expected(i), xnodes(i), DOUBLE_TOL);
	}
}

/*
Tests three points
*/
TEST(basis_tools_test_suite, test_1D_nodes_three_points){
	
	Kokkos::View<rtype*> xnodes("xnodes", 1);
	BasisTools::equidistant_nodes_1D_range(-1.0, 1.0, 3, xnodes);
	Kokkos::View<rtype*> expected("expected", 3);
	expected(0) = -1.0; 
	expected(1) = 0.0;
	expected(2) = 1.0;

	for (int i = 0; i < 3; i++){
		EXPECT_NEAR(expected(i), xnodes(i), DOUBLE_TOL);
	}
}