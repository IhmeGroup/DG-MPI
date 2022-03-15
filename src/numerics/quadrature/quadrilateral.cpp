#include "numerics/quadrature/quadrilateral.h"
#include "numerics/quadrature/segment.h"
#include "numerics/nodes.h"

namespace QuadrilateralQuadrature {

void get_quadrature_gauss_legendre(
    const int order,
    const int nq_1d,
    Kokkos::View<rtype**>::HostMirror& quad_pts,
    Kokkos::View<rtype*>::HostMirror& quad_wts) {

    // set up the segment quadrature points
    Kokkos::View<rtype**> quad_pts_1d("quad_pts_1d", nq_1d, 1);
    Kokkos::View<rtype*> quad_wts_1d("quad_wts_1d", nq_1d);

    host_view_type_2D h_quad_pts_1d = Kokkos::create_mirror_view(quad_pts_1d);
    host_view_type_1D h_quad_wts_1d = Kokkos::create_mirror_view(quad_wts_1d);

    // get quad nodes and wts for 1d views
    Nodes::get_gauss_legendre_segment_nodes(order,
        subview(h_quad_pts_1d, ALL(), 0));

    SegmentQuadrature::get_segment_weights_gl(order, h_quad_wts_1d);

    printf("order=%i\n", order);

    printf("nq1d=%i\n", nq_1d);

    for (int j = 0; j < nq_1d; j++) {
        for (int i = 0; i < nq_1d; i++) {
            quad_pts(j * nq_1d + i, 0) = h_quad_pts_1d(i, 0);
            quad_pts(j * nq_1d + i, 1) = h_quad_pts_1d(j, 0);
            quad_wts(j * nq_1d + i) = h_quad_wts_1d(i)
                * h_quad_wts_1d(j);
        }
    }
}

int get_gausslegendre_quadrature_order(const int order_,
    const int NDIMS){

    int qorder = SegmentQuadrature::get_gausslegendre_quadrature_order(order_, NDIMS);
    return qorder + 2; // add two for the non-constant jacobian that could exist
}

} // end namespace QuadrilateralQuadrature
