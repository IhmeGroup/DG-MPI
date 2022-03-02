#include <string>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "memory/memory_network.h"
#include "numerics/basis/basis.h"
#include "numerics/numerics_data.h"
#include "io/io_params.h"
#include "solver/base.h"

using std::cout, std::endl, std::string;

// Forward declaration
int main(int argc, char* argv[]);
void run_solver(toml::value& toml_input, MemoryNetwork& network);


int main(int argc, char* argv[]) {
    // Initialize memory network
    MemoryNetwork network(argc, argv);

    // Default input file name
    string toml_fname = "input.toml";
    // If a different name is specified, use that
    if (Utils::exist_option(argv, argv + argc, "-input")) {
        toml_fname = string(Utils::get_option(argv, argv + argc, "-input"));
    }
    // this call is increasing build time by a lot!!
    // TODO think of something to reduce this (Kihiro 2021/03/04)
    auto toml_input = toml::parse(toml_fname);
    const int dim = toml::find<int>(toml_input, "Physics", "dim");
    //cout << toml_input.report() << endl;

    // Run solver
    run_solver(toml_input, network);

    // Finalize memory network
    network.finalize();
}

void run_solver(toml::value& toml_input, MemoryNetwork& network) {
    // TODO
    int order = 1;
    // Get parameters related to the numerics
    auto numerics_params = Numerics::NumericsParams(toml_input, order);
    // Create mesh
    auto gbasis = Basis::Basis(numerics_params.gbasis, order);
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank,
            network.head_rank, gbasis);

    // Create physics
    

    // Create solver
    auto solver = Solver(toml_input, mesh, network, numerics_params);

    //// Precompute Helpers
    solver.precompute_matrix_helpers();

    // Finalize mesh
    mesh.finalize();
}
