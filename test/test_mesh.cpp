#include <gtest/gtest.h>
#include "mesh/mesh.h"

TEST(ReadMeshTest, BasicAssertions) {
	EXPECT_STRNE("hello", "world");
	EXPECT_EQ(7 * 6, 42);

	Mesh mesh;
	mesh.read_mesh("filename");

}
