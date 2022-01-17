//
// Created by kihiro on 12/29/20.
//

#include <string>
#include <iostream>
#include "toml11/toml.hpp"

using namespace std;


int main(int argc, char *argv[]) {

    cout << "Hello World!" << endl;


    // TODO: Below is some TOML-related stuff I figured we should keep for now,
    // since we'll be using it
    /*
    string toml_fname = "input.toml";
    if (Utils::exist_option(argv, argv + argc, "-input")) {
        toml_fname = string(Utils::get_option(argv, argv + argc, "-input"));
    }
    // this call is increasing build time by a lot!!
    // TODO think of something to reduce this (Kihiro 2021/03/04)
    auto toml_input = toml::parse(toml_fname);
    const int dim = toml::find<int>(toml_input, "Physics", "dim");

    // temporary hack to retrieve the equation type here - this needs rethinking
    auto eq_info = toml::find(toml_input, "Equation");
    const EquationType eq_t = get_equation_type(
        toml::find<std::string>(eq_info, "name"),
        toml::find_or<std::string>(eq_info, "inviscid_flux", "HLLC"),
        toml::find_or<std::string>(eq_info, "viscous_flux", "IP") );
    */
}
