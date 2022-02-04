#include "gtest/gtest.h"
#include <Kokkos_Core.hpp>
#include "memory/memory_network.h"


int main(int argc, char **argv) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    // Run GoogleTest
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
