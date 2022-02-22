#include <filesystem>
#include <iostream>
#include <string>
#include "memory/memory_network.h"

namespace fs = std::filesystem;

class MeshTestSuite {
    public:
        // Name of test suite
        string test_suite_name = "MeshTestSuite";
    private:
        // Memory network
        const MemoryNetwork network;

    public:
        // Constructor
        MeshTestSuite();
    private:
        void test_1();
};
