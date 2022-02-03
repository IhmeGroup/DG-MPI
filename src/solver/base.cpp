#include "solver/base.h"

Solver::Solver(const toml::value &input_file, Mesh& mesh, MemoryNetwork& network)
    : input_file{input_file}, mesh{mesh}, network{network} {
    // Size views
    Kokkos::resize(Uc, mesh.num_elems_part, 2, 2);
    Kokkos::resize(U_face, mesh.num_ifaces_part, 2, 2);
}
