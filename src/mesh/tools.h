#ifndef DG_MESH_TOOLS_H
#define DG_MESH_TOOLS_H

#include "common/defines.h"
#include "common/my_exceptions.h"
#include "mesh/mesh.h"
#include <Kokkos_Core.hpp>

namespace MeshTools {

	KOKKOS_INLINE_FUNCTION
	void elem_coords_from_elem_ID(Mesh& mesh, const int elem_ID,
		view_type_2D elem_coords);

	KOKKOS_INLINE_FUNCTION
	void ref_to_phys(Mesh& mesh, const int elem_ID, 
		view_type_2D basis_val, view_type_2D xphys, view_type_2D elem_coords);

	KOKKOS_INLINE_FUNCTION
	void get_element_volume(Mesh& mesh, const int elem_ID, 
		view_type_1D quad_wts, view_type_1D djac, rtype& vol,
		const member_type& member);


} // end namespace MeshTools

#include "mesh/tools.cpp"

#endif // end DG_MESH_TOOLS_H