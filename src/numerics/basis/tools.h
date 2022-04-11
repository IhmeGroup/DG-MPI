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
This function extracts the relevant node coordinates during the 
calculation of the normals to get the coordinates at the face from the 
element coordinates

Inputs:
-------
    dim: mesh dimension
    coord: element coordinates
    node_number: id containing the reference 
        coordinate number wrt the face nodes

Outputs:
--------
    extracted_coeffs: face coordinates extracted from the element coordinates
*/
template<typename ViewType2D, typename ViewType1D_int, 
typename ViewType1D_rtype> KOKKOS_INLINE_FUNCTION
void extract_node_coordinates(const unsigned dim, 
    const ViewType2D x_elem,
    const ViewType1D_int node_number,
    ViewType1D_rtype extracted_coeffs);


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
template<typename ViewType1D, typename ViewType2D, typename ViewType3D> KOKKOS_INLINE_FUNCTION
void get_element_jacobian(view_type_2D quad_pts,
    view_type_3D basis_ref_grad, ViewType3D jac, ViewType1D djac,
    ViewType3D ijac, ViewType2D elem_coords,
    const membertype& member);

template<typename ViewType1D, typename ViewType2D, typename ViewType3D> KOKKOS_INLINE_FUNCTION
void get_element_jacobian(view_type_2D quad_pts,
    view_type_3D basis_ref_grad, ViewType3D jac, ViewType1D djac,
    ViewType2D elem_coords,
    const membertype& member);


template<typename ViewType1D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
void get_inv_mass_matrices(view_type_1D quad_wts, view_type_2D basis_val,
    ViewType1D djac, ViewType2D iMM, const membertype& member);



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
void get_lagrange_basis_val_2D(host_view_type_2D_ls quad_pts,
    host_view_type_1D xnodes,
    int p, host_view_type_2D_ls basis_val);

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
void get_lagrange_basis_grad_2D(host_view_type_2D_ls quad_pts,
    host_view_type_1D xnodes,
    int p, host_view_type_3D_ls basis_ref_grad);

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
void get_lagrange_basis_val_3D(host_view_type_2D_ls quad_pts,
    host_view_type_1D xnodes,
    int p, host_view_type_2D_ls basis_val);

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
void get_lagrange_basis_grad_3D(host_view_type_2D_ls quad_pts,
    host_view_type_1D xnodes,
    int p, host_view_type_3D_ls basis_ref_grad);

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
void get_legendre_basis_val_2D(host_view_type_2D_ls quad_pts,
        const int p, host_view_type_2D_ls basis_val);

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
void get_legendre_basis_grad_2D(host_view_type_2D_ls quad_pts,
        const int p, host_view_type_3D_ls basis_ref_grad);

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
void get_legendre_basis_val_3D(host_view_type_2D_ls quad_pts,
        const int p, host_view_type_2D_ls basis_val);

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
void get_legendre_basis_grad_3D(host_view_type_2D_ls quad_pts,
        const int p, host_view_type_3D_ls basis_ref_grad);


} // end namespace BasisTools

#include "numerics/basis/tools.cpp"

#endif // DG_NUMERICS_BASIS_TOOLS_H
