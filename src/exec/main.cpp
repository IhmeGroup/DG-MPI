#include <string>
#include <iostream>
#include "toml11/toml.hpp"
#include "mesh/mesh.h"
#include "utils/utils.h"
#include "memory/memory_network.h"
// TODO: Uncomment
//#include "numerics/numerics_data.h"
#include "io/io_params.h"
#include "solver/base.h"

using std::cout, std::endl, std::string;


int main(int argc, char* argv[]) {
    // Initialize memory network
    MemoryNetwork network(argc, argv);

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
    //cout << toml_input.report() << endl;

    // TODO: This is just for testing
    // TODO: Uncomment
    //auto numerics_params = Numerics::NumericsParams(toml_input, 3);
    //auto solfile_params = SolutionFileParams(toml_input);

    // Create mesh
    auto mesh = Mesh(toml_input, network);

    // Create solver
    auto solver = Solver(toml_input, mesh, network);
}
