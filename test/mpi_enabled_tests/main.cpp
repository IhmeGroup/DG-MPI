#include "toml11/toml.hpp"
#include "memory/memory_network.h"
#include "test/mpi_enabled_tests/mesh/test_mesh.cpp"


int main(int argc, char* argv[]) {
    // Run mesh test suite
    MeshTestSuite(argc, argv);
}
