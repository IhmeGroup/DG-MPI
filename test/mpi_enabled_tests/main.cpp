#include "memory/memory_network.h"
#include "test/mpi_enabled_tests/memory/test_memory_network.h"
#include "test/mpi_enabled_tests/mesh/test_mesh.h"


int main(int argc, char* argv[]) {
    // Initialize memory network
    MemoryNetwork network(argc, argv);

    // Run memory network test suite
    MemoryTestSuite();
    // Run mesh test suite
    //MeshTestSuite();

    // Finalize memory network
    network.finalize();
}
