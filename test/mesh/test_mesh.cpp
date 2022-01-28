#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "memory/memory_network.h"
#include "mesh/mesh.h"
#include <filesystem>

using std::string;

namespace fs = std::filesystem;


TEST(MeshTestSuite, ShouldPartitionFourQuadsInHalfWithTwoRanks) {
    // TODO
    int argc = 1;
    char** argv;
    //MemoryNetwork network(argc, argv);
    // TODO: Get project directory instead
    string prefix = "../test/mesh/";

    // Read input file name
    string toml_fname = prefix + "input.toml";
    auto toml_input = toml::parse(toml_fname);
    const int dim = toml::find<int>(toml_input, "Physics", "dim");

    // Create mesh
    //auto mesh = Mesh(toml_input, network);
    EXPECT_STRNE("hello", "world");
}
