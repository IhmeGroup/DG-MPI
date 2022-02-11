#ifndef DG_SOLVER_HELPERS_H
#define DG_SOLVER_HELPERS_H

#include "common/defines.h"
#include "numerics/basis/basis.h"
#include "numerics/quadrature/tools.h"
#include <Kokkos_Core.hpp>

namespace VolumeHelpers {

struct VolumeHelperFunctor {

	// ~VolumeHelperFunctor() = default;
	VolumeHelperFunctor(Basis::Basis basis);

	KOKKOS_FUNCTION
    void operator()(const int ie) const;

 	void get_quadrature(Basis::Basis basis, 
 		const int order);

 	void get_reference_data(Basis::Basis basis, 
 		const int order);

    view_type_2D quad_pts;
    view_type_1D quad_wts;

    host_view_type_2D h_quad_pts;
    host_view_type_1D h_quad_wts;

    view_type_2D basis_val;
    view_type_3D basis_ref_grad;

    host_view_type_2D h_basis_val;
    host_view_type_3D h_basis_ref_grad;    
};

} // end namespace VolumeHelper

#include "solver/helpers.cpp"

#endif // DG_SOLVER_HELPERS_H