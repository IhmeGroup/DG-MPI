#include <filesystem>
#include <iostream>
#include <string>
#include "io/writer.h"
#include "memory/memory_network.h"
#include "mesh/mesh.h"
#include "../test/mpi_enabled_tests/mesh/test_mesh.h"

using std::string;

namespace fs = std::filesystem;

MeshTestSuite::MeshTestSuite() {
    // Run tests
    test_1();
};

void MeshTestSuite::test_1() {
    string test_case_name = "ShouldPartitionFourQuadsInHalfWithTwoRanks";

    // Location of input file
    string toml_fname = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/mesh/input.toml";
    // Read input file
    auto toml_input = toml::parse(toml_fname);

    // Create memory network
    auto network = MemoryNetwork();

    // Location of mesh file
    string mesh_file_name = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/mesh/quad_2x2.h5";
    // Create mesh
    auto gbasis = Basis::Basis();
    auto mesh = Mesh(toml_input, network.num_ranks, network.rank,
            network.head_rank, gbasis, mesh_file_name);
    // Create writer
    auto writer = Writer(mesh, network);
    // Cleanup
    mesh.finalize();
};
