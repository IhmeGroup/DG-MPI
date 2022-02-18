#include "solver/base.h"
#include "solver/helpers.h"
#include "numerics/basis/basis.h"
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
    mesh.gbasis = Basis::Basis(params.gbasis, params.gorder);

}


void Solver::precompute_matrix_helpers() {

    VolumeHelperFunctor functor(mesh, basis);

    Kokkos::parallel_for("volume helpers", team_policy( mesh.num_elems_part, Kokkos::AUTO )
        , functor);

}