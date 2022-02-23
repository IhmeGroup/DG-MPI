#ifndef DG_MESH_TOOLS_H
#define DG_MESH_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"
#include <Kokkos_Core.hpp>

namespace MeshTools {

	KOKKOS_INLINE_FUNCTION
	void elem_coords_from_elem_ID(const Mesh& mesh, const int elem_ID,
		scratch_view_2D_rtype elem_coords,
		const member_type& member);

	template<typename ViewType> KOKKOS_INLINE_FUNCTION
	void ref_to_phys(Mesh& mesh, const int elem_ID,
		view_type_2D basis_val, view_type_2D xphys,
		ViewType elem_coords,
		const member_type& member);

	KOKKOS_INLINE_FUNCTION
	void get_element_volume(Mesh& mesh, const int elem_ID,
		view_type_1D quad_wts, view_type_1D djac, rtype& vol,
		const member_type& member);


} // end namespace MeshTools

#include "mesh/tools.cpp"

#endif // end DG_MESH_TOOLS_H
