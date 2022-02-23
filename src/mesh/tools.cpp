#include "math/linear_algebra.h"

namespace MeshTools {

	KOKKOS_INLINE_FUNCTION
	void elem_coords_from_elem_ID(const Mesh& mesh, const int elem_ID,
		scratch_view_2D_rtype elem_coords, const member_type& member){

		auto node_IDs = Kokkos::subview(mesh.elem_to_node_IDs,
			elem_ID, Kokkos::ALL());

		for (int i = 0; i < elem_coords.extent(0); i++){
			for (int j = 0; j < elem_coords.extent(1); j++){
				elem_coords(i, j) = mesh.node_coords(node_IDs(i), j);
			}
		}
	}

	template<typename ScratchViewType2D, typename ViewType2D> KOKKOS_INLINE_FUNCTION
	void ref_to_phys(view_type_2D basis_val, ViewType2D xphys, 
		ScratchViewType2D elem_coords,
		const member_type& member){

		Math::cAxB_to_C(1., basis_val, elem_coords, xphys);
		printf("xphys%f\n:", xphys(0, 0));
	}

	template<typename ViewType1D> KOKKOS_INLINE_FUNCTION
	void get_element_volume(view_type_1D quad_wts, ViewType1D djac, rtype& vol,
		const member_type& member){

		int nq = djac.extent(0);
		Kokkos::parallel_reduce( Kokkos::TeamThreadRange( member, nq ),
			KOKKOS_LAMBDA ( const int iq, rtype &volupdate ) {

        volupdate += quad_wts( iq ) * djac( iq );
      }, vol);
	}

}
