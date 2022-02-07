#include <string>
#include <iostream>
#include "io/writer.h"
#include "memory/memory_network.h"
#include "mesh/mesh.h"
#include <filesystem>

using std::string;

namespace fs = std::filesystem;

class MeshTestSuite {
    public:
        // Name of test suite
        string test_suite_name = "MeshTestSuite";
    private:
        // Memory network
        const MemoryNetwork& network;

    public:
        // Constructor
        MeshTestSuite(int argc, char* argv[]) :
            network(MemoryNetwork(argc, argv)) {
            // Run tests
            mesh_test_1();
        };

    void mesh_test_1() {
        string test_case_name = "ShouldPartitionFourQuadsInHalfWithTwoRanks";

        // Location of input file
        string toml_fname = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/mesh/input.toml";
        // Read input file
        auto toml_input = toml::parse(toml_fname);

        // Location of mesh file
        string mesh_file_name = string(PROJECT_ROOT) + "/test/mpi_enabled_tests/mesh/quad_2x2.h5";
        // Create mesh
        auto mesh = Mesh(toml_input, network, mesh_file_name);
        // Create writer
        auto writer = Writer(mesh);
    };
};
