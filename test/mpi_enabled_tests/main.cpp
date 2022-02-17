#include "test/mpi_enabled_tests/memory/test_memory_network.cpp"
#include "test/mpi_enabled_tests/mesh/test_mesh.cpp"


int main(int argc, char* argv[]) {
    // Run memory network test suite
    MemoryTestSuite(argc, argv);
    // Run mesh test suite
    //MeshTestSuite(argc, argv);
}
