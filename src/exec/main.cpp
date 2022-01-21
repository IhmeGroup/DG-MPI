#include <string>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include <Kokkos_Core.hpp>

using namespace std;


int main(int argc, char *argv[]) {

    // Initialize Kokkos (This will need to be after MPI_Init)
    Kokkos::initialize(argc, argv);

    // Default input file name
    string toml_fname = "input.toml";
    // If a different name is specified, use that
    if (Utils::exist_option(argv, argv + argc, "-input")) {
        toml_fname = string(Utils::get_option(argv, argv + argc, "-input"));
    }
    // this call is increasing build time by a lot!!
    // TODO think of something to reduce this (Kihiro 2021/03/04)
    auto toml_input = toml::parse(toml_fname);
    const int dim = toml::find<int>(toml_input, "Physics", "dim");

    // Create mesh
    auto mesh = Mesh(toml_input);
}
