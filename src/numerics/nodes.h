#ifndef DG_NODES_H
#define DG_NODES_H

#include "common/defines.h"

#include <Kokkos_Core.hpp>

namespace Nodes {

using Kokkos::View;

/* Get the line equidistant nodes in reference space
	
Inputs:
-------
	order: order of polynomial containing nodes

Outputs:
--------
	xref - view to store coordinates [order + 1]

*/
KOKKOS_FUNCTION
void get_equidistant_nodes_segment(const int order, 
	View<rtype*> &xref);

/* Get the quadrilateral equidistant nodes in reference space
	
Inputs:
-------
	order: order of polynomial containing nodes

Outputs:
--------
	xref - view to store coordinates. [order + 1 * order + 1, 2]

*/
KOKKOS_FUNCTION
void get_equidistant_nodes_quadrilateral(const int order, 
	View<rtype**> &xref);


/* Get the hexahedron equidistant nodes in reference space
	
Inputs:
-------
	order: order of polynomial containing nodes

Outputs:
--------
	xref - view to store coordinates. [order + 1 * order + 1, 3]

*/
KOKKOS_FUNCTION
void get_equidistant_nodes_hexahedron(const int order, 
	View<rtype**> &xref);


/* Get the line Gauss-Legendre nodes in reference space

Inputs:
-------
    order: order of polynomial containing nodes

Outputs:
--------
    pts: view to store coordinates
*/
KOKKOS_FUNCTION
void get_gauss_legendre_segment_nodes(const int order, 
    View<rtype*>::HostMirror pts);


/* Get the line Gauss-Legendre-Lobatto (GLL) nodes in reference space

Inputs:
-------
    order: order of polynomial containing nodes

Outputs:
--------
    pts: view to store coordinates
*/
KOKKOS_FUNCTION
void get_gll_segment_nodes(const int order, 
    View<rtype*> pts);
} // end namespace Nodes


#endif // end DG_NODES_H