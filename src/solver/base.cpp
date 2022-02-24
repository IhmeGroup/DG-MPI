#include "solver/base.h"
#include "solver/helpers.h"
#include "numerics/basis/basis.h"
#include "common/defines.h"
#include <iostream>

using namespace VolumeHelpers;

Solver::Solver(const toml::value &input_file, Mesh& mesh, MemoryNetwork& network,
    Numerics::NumericsParams& params)
    : input_file{input_file}, mesh{mesh}, network{network}, params{params} {
    // Size views
    // Kokkos::resize(Uc, mesh.num_elems_part, 2, 2);
    // Kokkos::resize(U_face, mesh.num_ifaces_part, 2, 2);

    order = toml::find<int>(input_file, "Numerics", "order");
    basis = Basis::Basis(params.basis, order);
}


void Solver::precompute_matrix_helpers() {

    VolumeHelperFunctor functor(mesh, basis);

    // set scratch memory size for volume helpers
    int scratch_size = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim);
    functor.compute_inv_mass_matrices(scratch_size);
    Kokkos::fence();
    
    functor.compute_volume_helpers(scratch_size);
    Kokkos::fence();
    // Kokkos::View<rtype**> h_xphys("xphys", mesh.num_elems_part, functor.quad_pts.extent(0), mesh.dim);
    
    host_view_type_3D h_xphys = Kokkos::create_mirror_view(functor.x_elems);
    Kokkos::deep_copy(h_xphys, functor.x_elems);
    for (int k = 0; k < h_xphys.extent(0); k++){
    for (int i = 0; i < h_xphys.extent(1); i++){
        for (int j=0; j< h_xphys.extent(2); j++){
            printf("xphys(%i, %i, %i)=%f\n", k, i, j, h_xphys(k, i, j));
        }
    }
}

}
