#ifndef DG_QUADRATURE_HEXAHEDRON_H
#define DG_QUADRATURE_HEXAHEDRON_H

#include "common/defines.h"
#include "numerics/quadrature/segment.h"
#include <Kokkos_Core.hpp>

namespace HexahedronQuadrature {

using Kokkos::View;
using Kokkos::subview;
using Kokkos::ALL;

/*
Gets the Gauss Legendre quadrature points and weights
for a hexahedron shape

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


} // end namespace HexahedronQuadrature

#endif // DG_QUADRATURE_HEXAHEDRON_H
