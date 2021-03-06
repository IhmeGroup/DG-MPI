#ifndef DG_QUADRATURE_SEGMENT_H
#define DG_QUADRATURE_SEGMENT_H

#include "common/defines.h"
#include <Kokkos_Core.hpp>

namespace SegmentQuadrature {

using Kokkos::View;
using Kokkos::subview;
using Kokkos::ALL;


/*
Gets the Gauss Legendre segment weights

Inputs:
-------
    order: order of polynomial containing nodes

Outputs:
--------
    wts: view to store quadrature weights
*/
void get_segment_weights_gl(const int order,
    Kokkos::View<rtype*>::HostMirror wts);


/*
Gets the Gauss Legendre quadrature points and weights
for a segment shape

Inputs:
-------
    order: order of polynomial containing nodes
    nq: number of quadrature points

Outputs:
--------
    quad_pts: quadrature point coordinates [nq, ndims]
    quad_wts: quadrature weights [nq, 1]
*/
void get_quadrature_gauss_legendre(
    const int order,
    const int nq_1d,
    Kokkos::View<rtype**>::HostMirror& quad_pts,
    Kokkos::View<rtype*>::HostMirror& quad_wts);

int get_gausslegendre_quadrature_order(const int order_,
    const int NDIMS);

} // end namespace SegmentQuadrature

#endif // DG_QUADRATURE_SEGMENT_H
