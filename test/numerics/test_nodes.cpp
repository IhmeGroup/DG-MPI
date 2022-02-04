#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "numerics/nodes.h"

constexpr rtype DOUBLE_TOL = 1E-13;

/*
Tests the implementation of equidistant segments
*/
TEST(nodes_test_suite, test_segment_nodes){
	
	Kokkos::View<rtype*> xnodes("xnodes", 3);
	Nodes::get_equidistant_nodes_segment(2, xnodes);

	Kokkos::View<rtype*> expected("expected", 3);
	expected(0) = -1.; 
	expected(1) = 0.;
	expected(2) = 1.;

	for (int i = 0; i < 3; i++){
		EXPECT_NEAR(expected(i), xnodes(i), DOUBLE_TOL);
	}
}

/*
Tests the implementation of equidistant quadrilaterals
*/
TEST(nodes_test_suite, test_quadrilateral_nodes){
	
	Kokkos::View<rtype**> xnodes("xnodes", 9, 2);
	Nodes::get_equidistant_nodes_quadrilateral(2, xnodes);

	Kokkos::View<rtype**> expected("expected", 9, 2);
	expected(0, 0) = -1.; expected(0, 1) = -1.; 
	expected(1, 0) = 0.;  expected(1, 1) = -1.;
	expected(2, 0) = 1.;  expected(2, 1) = -1.;
	expected(3, 0) = -1.; expected(3, 1) = 0.; 
	expected(4, 0) = 0.;  expected(4, 1) = 0.;
	expected(5, 0) = 1.;  expected(5, 1) = 0.;
	expected(6, 0) = -1.; expected(6, 1) = 1.; 
	expected(7, 0) = 0.;  expected(7, 1) = 1.;
	expected(8, 0) = 1.;  expected(8, 1) = 1.;

	for (int i = 0; i < 9; i++){
		EXPECT_NEAR(expected(i, 0), xnodes(i, 0), DOUBLE_TOL);
		EXPECT_NEAR(expected(i, 1), xnodes(i, 1), DOUBLE_TOL);

	}
}

/*
Tests the implementation of equidistant hexahedron
*/
TEST(nodes_test_suite, test_hexahedral_nodes){
	
	Kokkos::View<rtype**> xnodes("xnodes", 27, 3);
	Nodes::get_equidistant_nodes_hexahedron(2, xnodes);

	Kokkos::View<rtype**> expected("expected", 27, 3);
	expected(0, 0) = -1.; expected(0, 1) = -1.; expected(0, 2) = -1.;
	expected(1, 0) = 0.;  expected(1, 1) = -1.; expected(1, 2) = -1.;
	expected(2, 0) = 1.;  expected(2, 1) = -1.; expected(2, 2) = -1.;
	expected(3, 0) = -1.; expected(3, 1) = 0.;  expected(3, 2) = -1.;
	expected(4, 0) = 0.;  expected(4, 1) = 0.;  expected(4, 2) = -1.;
	expected(5, 0) = 1.;  expected(5, 1) = 0.;  expected(5, 2) = -1.;
	expected(6, 0) = -1.; expected(6, 1) = 1.;  expected(6, 2) = -1.;
	expected(7, 0) = 0.;  expected(7, 1) = 1.;  expected(7, 2) = -1.;
	expected(8, 0) = 1.;  expected(8, 1) = 1.;  expected(8, 2) = -1.;

	expected(9, 0) = -1.; expected(9, 1) = -1.; expected(9, 2) = 0.;
	expected(10, 0) = 0.;  expected(10, 1) = -1.; expected(10, 2) = 0.;
	expected(11, 0) = 1.;  expected(11, 1) = -1.; expected(11, 2) = 0.;
	expected(12, 0) = -1.; expected(12, 1) = 0.;  expected(12, 2) = 0.;
	expected(13, 0) = 0.;  expected(13, 1) = 0.;  expected(13, 2) = 0.;
	expected(14, 0) = 1.;  expected(14, 1) = 0.;  expected(14, 2) = 0.;
	expected(15, 0) = -1.; expected(15, 1) = 1.;  expected(15, 2) = 0.;
	expected(16, 0) = 0.;  expected(16, 1) = 1.;  expected(16, 2) = 0.;
	expected(17, 0) = 1.;  expected(17, 1) = 1.;  expected(17, 2) = 0.;

	expected(18, 0) = -1.; expected(18, 1) = -1.; expected(18, 2) = 1.;
	expected(19, 0) = 0.;  expected(19, 1) = -1.; expected(19, 2) = 1.;
	expected(20, 0) = 1.;  expected(20, 1) = -1.; expected(20, 2) = 1.;
	expected(21, 0) = -1.; expected(21, 1) = 0.;  expected(21, 2) = 1.;
	expected(22, 0) = 0.;  expected(22, 1) = 0.;  expected(22, 2) = 1.;
	expected(23, 0) = 1.;  expected(23, 1) = 0.;  expected(23, 2) = 1.;
	expected(24, 0) = -1.; expected(24, 1) = 1.;  expected(24, 2) = 1.;
	expected(25, 0) = 0.;  expected(25, 1) = 1.;  expected(25, 2) = 1.;
	expected(26, 0) = 1.;  expected(26, 1) = 1.;  expected(26, 2) = 1.;
	for (int i = 0; i < 27; i++){
		EXPECT_NEAR(expected(i, 0), xnodes(i, 0), DOUBLE_TOL);
		EXPECT_NEAR(expected(i, 1), xnodes(i, 1), DOUBLE_TOL);
		EXPECT_NEAR(expected(i, 2), xnodes(i, 2), DOUBLE_TOL);

	}
}