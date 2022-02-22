#ifndef DG_NUMERICS_BASIS_TOOLS_H
#define DG_NUMERICS_BASIS_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"
#include <Kokkos_Core.hpp>

namespace BasisTools {

/*
This function gets the 1D equidistant coordinates in reference space

Inputs:
-------
	start: start of ref space (typically -1)
	stop:  end of ref space (typically 1)
	nnodes: num of nodes in 1D ref space

Outputs:
--------
	xnodes: coordinates of nodes in 1D ref space [nnodes]
*/
void equidistant_nodes_1D_range(rtype start, rtype stop, int nnodes,
	host_view_type_1D &xnodes);


/*
Evaluate the geometric Jacobian for a specified element

Inputs:
-------
	mesh: mesh object
	elem_ID: element ID
	quad_pts: coordinates of quadrature points

Outputs:
--------
	djac: determinant of the jacobian [nq]
	ijac: inverse of the jacobian [nq, ndims, ndims]
*/
KOKKOS_INLINE_FUNCTION
void get_element_jacobian(Mesh& mesh, const int elem_ID, view_type_2D quad_pts,
	view_type_3D basis_ref_grad, view_type_3D jac, view_type_1D djac,
	view_type_3D ijac, const member_type& member, view_type_2D elem_coords);


/*
Calculates the 1D Lagrange basis value

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_lagrange_basis_val_1D(const rtype &x,
	host_view_type_1D xnodes,
	int p, Kokkos::View<rtype*, Kokkos::LayoutStride>::HostMirror phi);

/*
Calculates the 1D Lagrange basis gradient

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_lagrange_basis_grad_1D(const rtype &x,
	host_view_type_1D xnodes,
	int p, Kokkos::View<rtype*, Kokkos::LayoutStride>::HostMirror gphi);

/*
Calculates the 2D Lagrange basis value

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_lagrange_basis_val_2D(host_view_type_2D quad_pts,
	host_view_type_1D xnodes,
	int p, host_view_type_2D basis_val);

/*
Calculates the 2D Lagrange basis gradient

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_lagrange_basis_grad_2D(host_view_type_2D quad_pts,
	host_view_type_1D xnodes,
	int p, host_view_type_3D basis_ref_grad);

/*
Calculates the 3D Lagrange basis value

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_lagrange_basis_val_3D(host_view_type_2D quad_pts,
	host_view_type_1D xnodes,
	int p, host_view_type_2D basis_val);

/*
Calculates the 3D Lagrange basis gradient

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_lagrange_basis_grad_3D(host_view_type_2D quad_pts,
	host_view_type_1D xnodes,
	int p, host_view_type_3D basis_ref_grad);

/*
Calculates the 1D Legendre basis value

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_legendre_basis_val_1D(const rtype &x, int p,
	Kokkos::View<rtype*, Kokkos::LayoutStride>::HostMirror phi);

/*
Calculates the 1D Legendre basis gradient

Inputs:
-------
	x: coordinate of current node [nq]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_legendre_basis_grad_1D(const rtype &x, int p,
	Kokkos::View<rtype*, Kokkos::LayoutStride>::HostMirror gphi);

/*
Calculates the 2D Legendre basis value

Inputs:
-------
	xq: coordinate of current node [nq, ndims]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_legendre_basis_val_2D(host_view_type_2D quad_pts,
		const int p, host_view_type_2D basis_val);

/*
Calculates the 2D Legendre basis gradient

Inputs:
-------
	xq: coordinate of current node [nq, ndims]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_legendre_basis_grad_2D(host_view_type_2D quad_pts,
		const int p, host_view_type_3D basis_ref_grad);

/*
Calculates the 3D Legendre basis value

Inputs:
-------
	xq: coordinate of current node [nq, ndims]
	p: order of polynomial space

Outputs:
--------
	basis_val: evaluated basis [nq, nb]
*/
void get_legendre_basis_val_3D(host_view_type_2D quad_pts,
		const int p, host_view_type_2D basis_val);

/*
Calculates the 3D Legendre basis gradient

Inputs:
-------
	xq: coordinate of current node [nq, ndims]
	p: order of polynomial space

Outputs:
--------
	basis_ref_grad: evaluated gradient of basis [nq, nb, ndims]
*/
void get_legendre_basis_grad_3D(host_view_type_2D quad_pts,
		const int p, host_view_type_3D basis_ref_grad);


} // end namespace BasisTools

#include "numerics/basis/tools.cpp"

#endif // DG_NUMERICS_BASIS_TOOLS_H
