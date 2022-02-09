#include "numerics/quadrature/hexahedron.h"
#include "numerics/nodes.h"

namespace HexahedronQuadrature {

// KOKKOS_FUNCTION
// void get_quadrature_gauss_legendre(
//     const int order,
//     int &nq,
//     View<rtype**>& quad_pts,
//     View<rtype*>& quad_wts) {

//     // set up the segment quadrature points
//     View<rtype**> quad_pts_1d("quad_pts_1d", 1, 1);
//     View<rtype*> quad_wts_1d("quad_wts_1d", 1);

//     Nodes::get_gauss_legendre_segment_nodes(order, 
//         subview(quad_pts_1d, ALL(), 0));

//     SegmentQuadrature::get_segment_weights_gl(order, quad_wts_1d);

//     int nq_1d = quad_wts_1d.extent(0);
//     nq = nq_1d * nq_1d * nq_1d;

//     // resize the hexahedron quadrature views
//     resize(quad_wts, nq);
//     resize(quad_pts, nq, 3);

//     for (int k = 0; k < nq_1d; k++){
//         for (int j = 0; j < nq_1d; j++) {
//             for (int i = 0; i < nq_1d; i++) {
//                 quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 0) = quad_pts_1d(i, 0);
//                 quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 1) = quad_pts_1d(j, 0);
//                 quad_pts(k * nq_1d * nq_1d + j * nq_1d + i, 2) = quad_pts_1d(k, 0);
//                 quad_wts(k * nq_1d * nq_1d + j * nq_1d + i) = quad_wts_1d(i) 
//                     * quad_wts_1d(j) * quad_wts_1d(k);
//             }
//         }
//     }
// }

} // end namespace HexahedronQuadrature