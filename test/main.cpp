#include "gtest/gtest.h"
#include "memory/memory_network.h"


int main(int argc, char **argv) {
    // Initialize memory network
    MemoryNetwork network(argc, argv);

    // Run GoogleTest
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
