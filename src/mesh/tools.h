#ifndef DG_MESH_TOOLS_H
#define DG_MESH_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"
#include <Kokkos_Core.hpp>

namespace MeshTools {

    template<typename ViewType2D> KOKKOS_INLINE_FUNCTION
    void elem_coords_from_elem_ID(const Mesh& mesh, const int elem_ID,
        ViewType2D elem_coords,
        const membertype& member);

    template<typename ScratchViewType2D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
    void ref_to_phys(view_type_2D basis_val, ViewType2D xphys,
        ScratchViewType2D elem_coords, const membertype& member);

    template<typename ViewType1D> KOKKOS_INLINE_FUNCTION
    void get_element_volume(view_type_1D quad_wts, ViewType1D djac, rtype& vol,
        const membertype& member);

    template<typename ViewType1D> inline
    rtype get_total_volume(const int num_elems_part, ViewType1D& vol_elems);


} // end namespace MeshTools

#include "mesh/tools.cpp"

#endif // end DG_MESH_TOOLS_H
