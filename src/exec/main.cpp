#include <string>
#include <vector>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "io/writer.h"
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

template<unsigned dim>
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
    const unsigned NDIMS = toml::find<unsigned>(toml_input, "Physics", "dim");
    if (NDIMS == 2){
        run_solver<2>(toml_input, network);
    } else if (NDIMS == 3){
        run_solver<3>(toml_input, network);
    }
    // Finalize memory network
    network.finalize();
}

template<unsigned dim>
void run_solver(toml::value& toml_input, MemoryNetwork& network) {
    // TODO: Add gorder from mesh file
    int gorder = 1;
    // Get parameters related to the numerics
    auto numerics_params = Numerics::NumericsParams(toml_input, gorder);
    // Create mesh
    auto gbasis = Basis::Basis(numerics_params.gbasis, gorder);
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank,
            network.head_rank, gbasis);

    const unsigned NDIMS = toml::find<unsigned>(toml_input, "Physics", "dim");

    auto solver = Solver<dim>(toml_input, mesh, network, numerics_params);

    // Read in InitialCondition data and copy it to the physics.IC_data view
    std::vector<rtype> IC_data_vec=toml::find<std::vector<rtype>>(toml_input, "InitialCondition", "data");
    assert(IC_data_vec.size()<=INIT_EX_PARAMS_MAX);

    Kokkos::resize(solver.physics.IC_data, IC_data_vec.size());
    host_view_type_1D h_IC_data = Kokkos::create_mirror_view(solver.physics.IC_data);
    // place initial condition data in host mirror view
    for (int i = 0; i<IC_data_vec.size(); i++){
        h_IC_data(i) = IC_data_vec[i];
    }
    // copy the initial condition data to the device
    Kokkos::deep_copy(solver.physics.IC_data, h_IC_data);

    // Precompute Helpers
    solver.precompute_matrix_helpers();
    printf("###################################################################\n");
    printf("Matrix helpers completed\n");
    printf("###################################################################\n");


    // Initialize the solution from the IC function.
    solver.init_state_from_fcn(mesh);
    printf("###################################################################\n");
    printf("Solution state initialized\n");
    printf("###################################################################\n");

    // ... we actually do the DG solve here
    solver.solve();

    // TODO: make this write a parallel hdf5 write
    // copy device solution coefficients to host
    solver.copy_from_device_to_host();
    int nb = solver.basis.get_num_basis_coeffs();
    int ns = solver.physics.get_NS();
    auto writer = Writer(mesh, network, solver.h_Uc, nb, ns, solver.time);
    printf("###################################################################\n");
    printf("Solution written to disk\n");
    printf("###################################################################\n");

    // Finalize mesh
    mesh.finalize();
}


