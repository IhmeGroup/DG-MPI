#include "memory/memory_network.h"
#include "test/mpi_enabled_tests/memory/test_memory_network.h"
#include "test/mpi_enabled_tests/mesh/test_mesh.h"
#include "test/mpi_enabled_tests/solver/test_helpers.h"

int main(int argc, char* argv[]) {
    // Initialize memory network
    MemoryNetwork network(argc, argv);

    // Run memory network test suite
    MemoryTestSuite();
    
    // Run mesh test suite
    //MeshTestSuite();

    // Run helpers test suite
    HelpersTestSuite();

    // Finalize memory network
    network.finalize();
}
