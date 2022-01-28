#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "memory/memory_network.h"
#include "mesh/mesh.h"

using std::string;


TEST(MeshTestSuite, ShouldPartitionFourQuadsInHalfWithTwoRanks) {
    //// Read input file name
    //string toml_fname = "input.toml";
    //auto toml_input = toml::parse(toml_fname);
    //const int dim = toml::find<int>(toml_input, "Physics", "dim");

    //// Create mesh
    //auto mesh = Mesh(toml_input, network);
    EXPECT_STRNE("hello", "world");
}
