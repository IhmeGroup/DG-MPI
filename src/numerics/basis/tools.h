#ifndef DG_NUMERICS_BASIS_TOOLS_H
#define DG_NUMERICS_BASIS_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"

#include "numerics/basis/basis.h"
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
	Kokkos::View<rtype*> &xnodes);

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
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype*> phi);

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
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype*> gphi);

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
void get_lagrange_basis_val_2D(Kokkos::View<const rtype**> quad_pts,
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype**> basis_val);

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
void get_lagrange_basis_grad_2D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype***> basis_ref_grad);

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
void get_lagrange_basis_val_3D(Kokkos::View<const rtype**> quad_pts,
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype**> basis_val);

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
void get_lagrange_basis_grad_3D(Kokkos::View<const rtype**> quad_pts, 
	Kokkos::View<const rtype*> xnodes,
	int p, Kokkos::View<rtype***> basis_ref_grad);

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
	Kokkos::View<rtype*> phi);

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
	Kokkos::View<rtype*> gphi);

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
void get_legendre_basis_val_2D(Kokkos::View<const rtype**> quad_pts, 
		const int p, Kokkos::View<rtype**> basis_val);

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
void get_legendre_basis_grad_2D(Kokkos::View<const rtype**> quad_pts, 
		const int p, Kokkos::View<rtype***> basis_ref_grad);

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
void get_legendre_basis_val_3D(Kokkos::View<const rtype**> quad_pts, 
		const int p, Kokkos::View<rtype**> basis_val);

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
void get_legendre_basis_grad_3D(Kokkos::View<const rtype**> quad_pts, 
		const int p, Kokkos::View<rtype***> basis_ref_grad);


} // end namespace BasisTools

#endif // DG_NUMERICS_BASIS_TOOLS_H