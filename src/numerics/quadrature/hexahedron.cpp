#include "numerics/quadrature/hexahedron.h"
#include "numerics/nodes.h"

namespace HexahedronQuadrature {

void get_quadrature_gauss_legendre(
    const int order,
    const int nq_1d,
    Kokkos::View<rtype**>::HostMirror& quad_pts,
    Kokkos::View<rtype*>::HostMirror& quad_wts) {

    // set up the segment quadrature points
    View<rtype**> quad_pts_1d("quad_pts_1d", nq_1d, 1);
    View<rtype*> quad_wts_1d("quad_wts_1d", nq_1d);

	host_view_type_2D h_quad_pts_1d = Kokkos::create_mirror_view(quad_pts_1d);
    host_view_type_1D h_quad_wts_1d = Kokkos::create_mirror_view(quad_wts_1d);

    // get quad nodes and wts for 1d views
    Nodes::get_gauss_legendre_segment_nodes(order, 
        subview(h_quad_pts_1d, ALL(), 0));

    SegmentQuadrature::get_segment_weights_gl(order, h_quad_wts_1d);

    for (int k = 0; k < nq_1d; k++){
        for (int j = 0; j < nq_1d; j++) {
            for (int i = 0; i < nq_1d; i++) {
                quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 0) = h_quad_pts_1d(i, 0);
                quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 1) = h_quad_pts_1d(j, 0);
                quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 2) = h_quad_pts_1d(k, 0);
                quad_wts(k * nq_1d * nq_1d + j * nq_1d + i) = h_quad_wts_1d(i) 
                    * h_quad_wts_1d(j) * h_quad_wts_1d(k);
            }
        }
    }
}

} // end namespace HexahedronQuadrature