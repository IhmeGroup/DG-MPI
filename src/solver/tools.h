#ifndef DG_SOLVER_TOOLS_H
#define DG_SOLVER_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include <Kokkos_Core.hpp>

namespace SolverTools {

inline
void calculate_volume_flux_integral(const int num_elems, view_type_3D basis_ref_grad,
    view_type_4D F_quad, view_type_3D res);

inline
void calculate_face_flux_integral(const int num_ifaces, view_type_2D basis_val, 
    view_type_3D F_quad, view_type_3D res);

template<typename ViewType_iMM, typename ViewType2D, typename ViewType1D_djac, 
typename ViewType1D_quadwts, typename ViewType2D_f, typename ViewType2D_state>
KOKKOS_INLINE_FUNCTION
void L2_projection(ViewType_iMM iMM, ViewType2D basis_val, ViewType1D_djac djac,
    ViewType1D_quadwts quad_wts, ViewType2D_f f, ViewType2D_state U,
    const membertype& member);

template<typename ViewType_iMM, typename ViewType_res> inline
void mult_inv_mass_matrix(const rtype dt, const ViewType_iMM iMM_elems,
    const ViewType_res res, ViewType_res dU);



} // end namespace MeshTools

#include "solver/tools.cpp"

#endif // end DG_SOLVER_TOOLS_H
