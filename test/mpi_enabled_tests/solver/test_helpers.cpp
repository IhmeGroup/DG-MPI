#include <string>
#include <iostream>
#include "toml11/toml.hpp"

#include "memory/memory_network.h"
#include "numerics/basis/basis.h"
#include "numerics/numerics_data.h"
#include "mesh/mesh.h"
#include "solver/base.h"

#include "../test/mpi_enabled_tests/solver/test_helpers.h"

using std::string;

HelpersTestSuite::HelpersTestSuite(){
    // run tests
    test_1();
}


void HelpersTestSuite::test_1(){
    string test_case_name = "VolumeHelperData";

    auto network = MemoryNetwork();

    // Location of input file
    string toml_fname = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/solver/input.toml";
    // Read input file
    auto toml_input = toml::parse(toml_fname);

    // Location of mesh file
    string mesh_file_name = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/solver/quad_2x2.h5";

    // set gorder = 1
    int order = 1;
    // Get parameters related to the numerics
    auto numerics_params = Numerics::NumericsParams(toml_input, order);
    // Create mesh
    auto gbasis = Basis::Basis(numerics_params.gbasis, order);
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank, network.head_rank, 
        gbasis, mesh_file_name);


    //TODO: UPDATE THIS TEST
    // Create solver
    // auto solver = Solver(toml_input, mesh, network, pnumerics_params);

    //// Precompute Helpers
    // solver.precompute_matrix_helpers();



}