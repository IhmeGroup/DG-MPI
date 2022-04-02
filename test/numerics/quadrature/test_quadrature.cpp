#include <cstdlib>
#include <math.h>
#include "gtest/gtest.h"
#include "common/defines.h"
#include "numerics/nodes.h"
#include "numerics/basis/basis.h"
#include "numerics/basis/shape.h"
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
	int nq_1d;
	rtype segment_length = 2.0;
	rtype quadrilateral_area = 4.0;
	rtype hexahedron_area = 8.0;
    
    // 1D Line Segment
    // instantiate the shape
    auto shape = Basis::Shape(ShapeType::Segment);
    int NDIMS = 1;

	for (int order = 1; order < 20; order++){

		View<rtype**> d_quad_pts("quad_pts", 1, 1);
		View<rtype*> d_quad_wts("quad_wts", 1);

        int qorder = shape.get_quadrature_order(order);
        QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS, nq_1d, nq);

        Kokkos::resize(d_quad_pts, nq, NDIMS);
        Kokkos::resize(d_quad_wts, nq);

        host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(d_quad_pts);
        host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(d_quad_wts);

		shape.get_quadrature_data(qorder,
    		nq_1d, h_quad_pts, h_quad_wts);

		int nwts = h_quad_wts.extent(0);

		rtype segment_sum = 0.;
		for (int i = 0; i < nwts; i++){
			segment_sum += h_quad_wts(i);
		}

		EXPECT_NEAR(segment_sum, segment_length, DOUBLE_TOL);
    }

    // 2D Quadrilateral
    // instantiate the shape
    shape = Basis::Shape(ShapeType::Quadrilateral);
    NDIMS = 2;

    for (int order = 1; order < 20; order++){

        View<rtype**> d_quad_pts("quad_pts", 1, 1);
        View<rtype*> d_quad_wts("quad_wts", 1);

        int qorder = shape.get_quadrature_order(order);
        QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS, nq_1d, nq);

        Kokkos::resize(d_quad_pts, nq, NDIMS);
        Kokkos::resize(d_quad_wts, nq);

        host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(d_quad_pts);
        host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(d_quad_wts);

        shape.get_quadrature_data(qorder,
            nq_1d, h_quad_pts, h_quad_wts);

        int nwts = h_quad_wts.extent(0);

        rtype quad_sum = 0.;
        for (int i = 0; i < nwts; i++){
            quad_sum += h_quad_wts(i);
        }

        EXPECT_NEAR(quad_sum, quadrilateral_area, DOUBLE_TOL);
    }

    // 3D Hexahedron
    // instantiate the shape
    shape = Basis::Shape(ShapeType::Hexahedron);
    NDIMS = 3;

    for (int order = 1; order < 20; order++){

        View<rtype**> d_quad_pts("quad_pts", 1, 1);
        View<rtype*> d_quad_wts("quad_wts", 1);

        int qorder = shape.get_quadrature_order(order);
        QuadratureTools::get_number_of_quadrature_points(qorder, NDIMS, nq_1d, nq);

        Kokkos::resize(d_quad_pts, nq, NDIMS);
        Kokkos::resize(d_quad_wts, nq);

        host_view_type_2D h_quad_pts = Kokkos::create_mirror_view(d_quad_pts);
        host_view_type_1D h_quad_wts = Kokkos::create_mirror_view(d_quad_wts);

        shape.get_quadrature_data(qorder,
            nq_1d, h_quad_pts, h_quad_wts);

        int nwts = h_quad_wts.extent(0);

        rtype hex_sum = 0.;
        for (int i = 0; i < nwts; i++){
            hex_sum += h_quad_wts(i);
        }

        EXPECT_NEAR(hex_sum, hexahedron_area, DOUBLE_TOL);
    }
}

/* Fix this test to mean someting */
TEST(quadrature_test_suite, quad_rule){

    Basis::Basis basis(enum_from_string<BasisType>("LagrangeSeg"), 2);

    int qorder = basis.shape.get_quadrature_order(2);

}