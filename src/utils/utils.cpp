#include <cstdlib>
#include <iostream>
#include <string>
#include "toml11/toml.hpp"
#include "common/my_exceptions.h"
#include "utils/utils.h"

namespace Utils {

using namespace std;

string get_file_content(const char *fname) {
    stringstream msg;
    string line;
    ifstream file;
    file.open(fname);
    msg << string(80, '=') << endl;
    while (getline(file, line)) {
        msg << line << endl;
    }
    msg << string(80, '=') << endl << endl;
    return msg.str();
}

// TODO update this check (Kihiro 2021/03/04)
void check_input_file(const toml::value &input) {
    if (!input.contains("Mesh") ||
        !input.at("Mesh").contains("file")) {
        throw InputException("Mesh information incomplete");
    }
    if (!input.contains("Numerics") ||
        !input.at("Numerics").contains("basis") ||
        !input.at("Numerics").contains("order")) {
        throw InputException("Numerics information incomplete");
    }
    if (!input.contains("Output") ||
        !input.at("Output").contains("prefix")) {
        throw InputException("Output information incomplete");
    }
    if (!input.contains("Stepper") ||
        !input.at("Stepper").contains("start") ||
        !input.at("Stepper").contains("end") ||
        !input.at("Stepper").contains("timestep")) {
        throw InputException("Stepper information incomplete");
    }
    if (!input.contains("Physics") ||
        !input.at("Physics").contains("dim")) {
        throw InputException("Physics information incomplete");
    }
    if (!input.contains("Equation") ||
        !input.at("Equation").contains("name")) {
        throw InputException("Equation information incomplete");
    }
    if (input.contains("Boundaries")) {
        vector<string> BFG_names = toml::find<vector<string>>(input, "Boundaries", "names");
        for (string bname: BFG_names) {
            if (!input.at("Boundaries").contains(bname) ||
                !input.at("Boundaries").at(bname).contains("type")) {
                throw InputException("Boundary information incomplete");
            }
        }
    }
}

/*! \brief Helper function to create a directory
 *
 * @param prefix path containing at least one /, otherwise nothing is created
 *               the directory created is the sub-string prior to the last /
 */
static void create_directory(const std::string &prefix) {
    string dir = prefix.substr(0, prefix.find_last_of('/'));
    if (dir.size() < prefix.size()) {
        string cmd = "mkdir -p " + dir;
        int status = system(cmd.c_str());
        if (status != 0) {
            stringstream msg;
            msg << "Could not create directory " << dir;
            throw SystemCallException(msg.str());
        }
    }
}

void create_required_directories(const toml::value &input_info) {
    // this one should be of the form "solutions/prefix"
    // TODO make solutions default as restarts... (Kihiro 2021/03/04)
    create_directory(toml::find<string>(input_info, "Output", "prefix"));
    create_directory("restarts/");
    // for Legion logging
    // the location where those logs are placed is however controlled at the command line level
    create_directory("logs/");
    // text_outputs is created only if text outputs are requested
    if (input_info.contains("Diagnosis") ||
        input_info.contains("Error") ||
        input_info.contains("time_run") ||
        input_info.contains("Statistics")) {

        create_directory("text_outputs/");
    }
}

string get_fname_with_iter(const char *prefix, const char *suffix, const int iter) {
    stringstream name;
    name << prefix
         <<  "_"  << setfill('0') << std::setw(8) << iter
         << suffix;
    return name.str();
}

string get_fname_with_iter_srID(
    const char *prefix,
    const char *suffix,
    const unsigned iter,
    const unsigned srID) {

    stringstream name;
    name << prefix << setfill('0')
         <<  "_"  << std::setw(8) << iter
         <<  "_" << std::setw(3) << srID << suffix;
    return name.str();
}

char* get_option(char ** begin, char ** end, const std::string & option) {
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool exist_option(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

void print_header() {
    auto text = "Hello world!";
    std::cout << text << std::endl;
}

} // namespace Utils
