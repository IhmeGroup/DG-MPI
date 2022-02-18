#include "math/linear_algebra.h"

namespace MeshTools {

	KOKKOS_INLINE_FUNCTION
	void elem_coords_from_elem_ID(Mesh& mesh, const int elem_ID,
		view_type_2D elem_coords){
		
		auto node_IDs = Kokkos::subview(mesh.elem_to_node_IDs, 
			elem_ID, Kokkos::ALL());

		for (int i = 0; i < mesh.num_nodes_per_elem; i++){
			for (int j = 0; j < mesh.dim; j++){
				elem_coords(i, j) = mesh.node_coords(node_IDs(i), j);
			}
		}
	}

	KOKKOS_INLINE_FUNCTION
	void ref_to_phys(Mesh& mesh, const int elem_ID, 
		view_type_2D basis_val, view_type_2D xphys){

		Kokkos::View<rtype**> elem_coords("elem_coords", 
			mesh.num_nodes_per_elem, mesh.dim);
		elem_coords_from_elem_ID(mesh, elem_ID, elem_coords);

		Math::cAxB_to_C(1., basis_val, elem_coords, xphys);

	}

}