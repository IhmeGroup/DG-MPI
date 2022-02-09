#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "numerics/nodes.h"
#include "numerics/basis/basis.h"
#include "numerics/quadrature/segment.h"
#include "numerics/quadrature/quadrilateral.h"
#include "numerics/quadrature/hexahedron.h"

constexpr rtype DOUBLE_TOL = 1E-13;

using Kokkos::View;
/*
Tests the implementation of equidistant segments
*/
KOKKOS_FUNCTION
TEST(quadrature_test_suite, test_weights_sum_to_area){
	
	int nq;
	rtype segment_length = 2.0;
	rtype quadrilateral_area = 4.0;
	rtype hexahedron_area = 8.0;

	for (int order = 1; order < 20; order++){

		View<rtype**> quad_pts("quad_pts", 1, 1);
		View<rtype*> quad_wts("quad_wts", 1);

		// 1D Line Segment
		SegmentQuadrature::get_quadrature_gauss_legendre(order,
    		nq, quad_pts, quad_wts);

		int nwts = quad_wts.extent(0);

		rtype segment_sum = 0.;
		for (int i = 0; i < nwts; i++){
			segment_sum += quad_wts(i);
		}

		EXPECT_NEAR(segment_sum, segment_length, DOUBLE_TOL);

		// 2D Quadrilateral
		QuadrilateralQuadrature::get_quadrature_gauss_legendre(order,
    		nq, quad_pts, quad_wts);

		nwts = quad_wts.extent(0);

		rtype quadrilateral_sum = 0.;
		for (int i = 0; i < nwts; i++){
			quadrilateral_sum += quad_wts(i);
		}
		EXPECT_NEAR(quadrilateral_sum, quadrilateral_area, DOUBLE_TOL);

		// 3D Hexahedron
		HexahedronQuadrature::get_quadrature_gauss_legendre(order,
    		nq, quad_pts, quad_wts);

		nwts = quad_wts.extent(0);
		
		rtype hexahedron_sum = 0.;
		for (int i = 0; i < nwts; i++){
			hexahedron_sum += quad_wts(i);
		}
		EXPECT_NEAR(hexahedron_sum, hexahedron_area, DOUBLE_TOL);


	}
}

/* Fix this test to mean someting */
TEST(quadrature_test_suite, quad_rule){

    Basis::Basis basis(enum_from_string<BasisType>("LagrangeSeg"), 2);

    int qorder = basis.shape.get_quadrature_order(2);

    std::cout<<qorder<<std::endl;
}