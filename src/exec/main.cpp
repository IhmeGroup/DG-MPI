#include <string>
#include <vector>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "memory/memory_network.h"
#include "numerics/basis/basis.h"
#include "physics/base/base.h"
#include "numerics/numerics_data.h"
#include "io/io_params.h"
#include "solver/base.h"

#include <Kokkos_Core.hpp>

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
    std::string phys = toml::find<std::string>(toml_input, "Physics", "name");
    auto physics_type = enum_from_string<PhysicsType>(phys.c_str());
    // auto physics_type = PhysicsType::Euler;
    // auto physics = Physics::Physics(physics_type, mesh.dim);

    // Create solver
    auto solver = Solver(toml_input, mesh, network, numerics_params,
        physics_type);

    const auto IC_name = toml::find<std::string>(toml_input, "InitialCondition", "name");
    // const auto IC_data = toml::find<rtype[10]>(toml_input, "InitialCondition", "data");

    // Read in InitialCondition data and copy it to the physics.IC_data view
    std::vector<rtype> IC_data_vec=toml::find<std::vector<rtype>>(toml_input, "InitialCondition", "data");
    assert(IC_data_vec.size()<=INIT_EX_PARAMS_MAX);

    Kokkos::resize(solver.physics.IC_data, IC_data_vec.size());
    host_view_type_1D h_IC_data = Kokkos::create_mirror_view(solver.physics.IC_data);

    for (int i = 0; i<IC_data_vec.size(); i++){
        h_IC_data(i) = IC_data_vec[i];
    }
    Kokkos::deep_copy(solver.physics.IC_data, h_IC_data);

    // solver.physics.IC_name = IC_name.c_str();
    // Set the initial condition function
    // solver.physics.set_IC(solver.physics, IC_name);

    // Precompute Helpers
    solver.precompute_matrix_helpers();


    // Initialize the solution from the IC function.
    solver.init_state_from_fcn(mesh);

    // ... we actually do the DG solve here

    // Finalize mesh
    mesh.finalize();
}
