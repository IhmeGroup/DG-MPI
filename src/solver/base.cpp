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

    VolumeHelperFunctor functor;

    // need to get the sizes of things to pass into scratch memory
    int nb = basis.get_num_basis_coeffs();
    int nq_1d; int nq;
    int qorder = basis.shape.get_quadrature_order(2 * basis.get_order());
    QuadratureTools::get_number_of_quadrature_points(qorder, mesh.dim,
            nq_1d, nq);

    printf("nq=%i", nq);
    printf("nb=%i", nb);
    // set scratch memory size for iMM helper
    int scratch_size_iMM = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim)
        + scratch_view_2D_rtype::shmem_size(nb, nb) + scratch_view_2D_rtype::shmem_size(nq, nb);
    int scratch_size_vol = scratch_view_2D_rtype::shmem_size(mesh.num_nodes_per_elem, mesh.dim);

    printf("scratch_iMM=%i\n", scratch_size_iMM);
    printf("scratch_vol=%i\n", scratch_size_vol);
    functor.compute_inv_mass_matrices(scratch_size_iMM, mesh, basis);
    Kokkos::fence();


    functor.compute_volume_helpers(scratch_size_vol, mesh, basis);
    Kokkos::fence();

    host_view_type_3D h_xphys = Kokkos::create_mirror_view(functor.x_elems);
    host_view_type_3D h_iMM_elems = Kokkos::create_mirror_view(functor.iMM_elems);

    Kokkos::deep_copy(h_xphys, functor.x_elems);
    Kokkos::deep_copy(h_iMM_elems, functor.iMM_elems);

    for (int k = 0; k < h_xphys.extent(0); k++){
    for (int i = 0; i < h_xphys.extent(1); i++){
        for (int j=0; j< h_xphys.extent(2); j++){
            printf("xphys(%i, %i, %i)=%f\n", k, i, j, h_xphys(k, i, j));
        }
    }
}

    for (int k = 0; k < h_iMM_elems.extent(0); k++){
    for (int i = 0; i < h_iMM_elems.extent(1); i++){
        for (int j=0; j< h_iMM_elems.extent(2); j++){
            printf("iMM_elems(%i, %i, %i)=%f\n", k, i, j, h_iMM_elems(k, i, j));
        }
    }
}

}
