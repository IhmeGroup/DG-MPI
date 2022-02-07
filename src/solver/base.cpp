#include "solver/base.h"
#include "solver/helpers.h"

// using namespace VolumeHelpers;

Solver::Solver(const toml::value &input_file, Mesh& mesh, MemoryNetwork& network)
    : input_file{input_file}, mesh{mesh}, network{network} {
    // Size views
    Kokkos::resize(Uc, mesh.num_elems_part, 2, 2);
    Kokkos::resize(U_face, mesh.num_ifaces_part, 2, 2);
}


void Solver::precompute_matrix_helpers() {

    VolumeHelpers::VolumeHelperFunctor functor();

    int num_elements = 1000;

    Kokkos::parallel_for(num_elements, functor);

}