#ifndef DG_QUADRATURE_SEGMENT_H
#define DG_QUADRATURE_SEGMENT_H

#include "common/defines.h"
#include <Kokkos_Core.hpp>

namespace Quadrature {

using Kokkos::View;
/*
Gets the Gauss Legendre segment weights

Inputs:
-------
    order: order of polynomial containing nodes

Outputs:
--------
    wts: view to store quadrature weights
*/
KOKKOS_FUNCTION
void get_segment_weights_gl(const int order, 
    View<rtype*> &wts);

// void get_gaussian_quadrature_1D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);
// void get_gaussian_quadrature_2D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);
// void get_gaussian_quadrature_3D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);
// void get_gaussian_quadrature_tri(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);

// void get_line_weights_gll(const unsigned order, std::vector<rtype> &wts);
// void get_gauss_lobatto_quadrature_1D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);
// void get_gauss_lobatto_quadrature_2D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);
// void get_gauss_lobatto_quadrature_3D(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);

// unsigned get_num_points_dunavant(const unsigned order);
// void get_dunavant_quad_points(const unsigned order, std::vector<rtype> &pts);
// void get_dunavant_quad_weights(const unsigned order, std::vector <rtype> &wts);
// void get_dunavant_quadrature(
//     const unsigned order,
//     unsigned &nq,
//     std::vector<rtype> &quad_pts,
//     std::vector<rtype> &quad_wts);

} // end namespace Quadrature

#endif //DG_QUADRATURE_H