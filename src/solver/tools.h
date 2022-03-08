#ifndef DG_SOLVER_TOOLS_H
#define DG_SOLVER_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include <Kokkos_Core.hpp>

namespace SolverTools {

template<typename ViewType_iMM, typename ViewType2D, typename ViewType1D_djac, 
    typename ViewType1D_quadwts, typename ViewType2D_f, typename ViewType2D_state>
KOKKOS_INLINE_FUNCTION
void L2_projection(ViewType_iMM iMM, ViewType2D basis_val, ViewType1D_djac djac,
    ViewType1D_quadwts quad_wts, ViewType2D_f f, ViewType2D_state U,
    const member_type& member);


} // end namespace MeshTools

#include "solver/tools.cpp"

#endif // end DG_SOLVER_TOOLS_H
