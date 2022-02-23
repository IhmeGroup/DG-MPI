#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include "common/defines.h"
#include "mesh/mesh.h"
#include "mesh/tools.h"
#include "numerics/basis/basis.h"
#include "numerics/basis/tools.h"
#include "numerics/quadrature/tools.h"
#include <Kokkos_Core.hpp>

namespace VolumeHelpers {

struct VolumeHelperFunctor {

	VolumeHelperFunctor(Mesh mesh, Basis::Basis basis);

	KOKKOS_INLINE_FUNCTION
    void operator()(const member_type& member) const;

 	void get_quadrature(Basis::Basis basis,
 		const int order);

 	void get_reference_data(Basis::Basis basis,
        Basis::Basis gbasis,
 		const int order);

    void allocate_views(Mesh& mesh, const int num_elems);

    Mesh mesh;

    view_type_2D quad_pts;
    view_type_1D quad_wts;

    host_view_type_2D h_quad_pts;
    host_view_type_1D h_quad_wts;

    view_type_2D basis_val;
    view_type_3D basis_ref_grad;

    host_view_type_2D h_basis_val;
    host_view_type_3D h_basis_ref_grad;

    view_type_2D gbasis_val;
    view_type_3D gbasis_ref_grad;

    host_view_type_2D h_gbasis_val;
    host_view_type_3D h_gbasis_ref_grad;

    view_type_4D jac_elems;
    view_type_2D djac_elems;
    view_type_4D ijac_elems;

    view_type_3D x_elems;
    view_type_1D vol_elems;
};


} // end namespace VolumeHelper

#include "solver/helpers.cpp"

#endif // DG_SOLVER_HELPERS_H
