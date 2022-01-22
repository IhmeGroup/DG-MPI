#include <iostream>
#include "io/io_params.h"
//#include "equations/equation_data.h"
//#include "physics/physics_data.h"

using namespace std;

BoundaryOutputParams::BoundaryOutputParams(
    const toml::value &input_info,
    const std::vector<std::string> &bfg_names) {
//
//    // check if there is a Diagnosis section in the input file
//    if (input_info.contains("Diagnosis")) {
//        // iterate over boundary face groups
//        for (unsigned ibfg = 0; ibfg < bfg_names.size(); ibfg++) {
//            const string &name = bfg_names[ibfg];
//            if (input_info.at("Diagnosis").contains(name)) {
//                // store the index of the boundary face group, nout is initialized to zero
//                idxs[nout] = ibfg;
//                // store the name for each variable for which diagnosis was requested
//                vector<std::string> _var_names = toml::find<vector<string>>(
//                    input_info.at("Diagnosis").at(name), "variables");
//                assert(_var_names.size() < N_VAR_BDIAG_MAX);
//                nvar[nout] = _var_names.size();
//                for (unsigned ivar = 0; ivar < _var_names.size(); ivar++) {
//                    sprintf(var_names[nout][ivar], "%s", _var_names[ivar].c_str());
//                }
//                // store the name of the current boundary face group
//                sprintf(boundary_names[nout], "%s", name.c_str());
//                nout++;
//            }
//        }
//    }
}

SolutionFileParams::SolutionFileParams(const toml::value &input) {
    string name;
    auto output_info = toml::find(input, "Output");

    // Read basis information
    name = toml::find<std::string>(input, "Numerics", "basis");
    // TODO
    //const BasisType basis_t = basis_type_from_string(name.c_str());

    name = toml::find_or<string>(output_info, "node_type", "Equidistant");
    node_t = NodeType::Equidistant;//TODO NodeType::_from_string(name.c_str());
    if (output_info.contains("order")) {
        order = (unsigned) toml::find<int>(output_info, "order");
    }
    else {
        // by default, the order is the same as the solution's order
        order = (unsigned) toml::find<int>(input, "Numerics", "order");
    }
//    EnumMap::get_solfile_params(basis_t, order, nppp, nppsc, nppscn);
//    // -1 means only write the first and final solution
//    interval = toml::find_or<int>(output_info, "interval", -1);
//    std::string prefix_ = toml::find<std::string>(input, "Output", "prefix");
//    assert(prefix_.size() < FILE_NAME_LEN_MAX);
//    prefix_.copy(prefix, prefix_.size() + 1);
//    prefix[prefix_.size()] = '\0';
//    auto varsname_ = toml::find<std::vector<std::string>>(output_info, "variables");
//    nvar = 0;
//    for (std::string varname: varsname_) {
//        PhysicalVariable var = EnumMap::get_physical_variable(varname);
//        if (var == PhysicalVariable::DUMMY) {
//            std::cout << "Variable " << varname << " unrecongnized. Ignoring it." << std::endl;
//            continue;
//        }
//        vars[nvar] = var;
//        sprintf(varsname[nvar], "%s", varname.c_str());
//        nvar++;
//    }
}

VolumeDiagnosisParams::VolumeDiagnosisParams(const toml::value &input) {
//    if (input.contains("Diagnosis")) {
//        interval = toml::find_or<int>(input.at("Diagnosis"), "interval", 1);
//        vector<string> varsname_ = toml::find<std::vector<std::string>>(
//            input.at("Diagnosis"), "variables");
//        assert(varsname_.size() <= N_VAR_OUTPUT_MAX);
//        nvar = varsname_.size();
//        for (unsigned ivar = 0; ivar < varsname_.size(); ivar++) {
//            sprintf(varsname[ivar], "%s", varsname_[ivar].c_str());
//        }
//        if (input.at("Diagnosis").contains("quad_order")) {
//            quad_order = (unsigned) toml::find<int>(input.at("Diagnosis"), "quad_order");
//            quad_order_same = false;
//        }
//        else {
//            quad_order_same = true;
//        }
//    }
}

StatsParams::StatsParams(const toml::value &input) {
//    if (input.contains("Statistics")) {
//        auto stats_input = toml::find(input, "Statistics");
//        interval = toml::find_or<int>(stats_input, "interval", -1);
//        if (stats_input.contains("quad_order")) {
//            quad_order = (unsigned) toml::find<int>(stats_input, "quad_order");
//            quad_order_same = false;
//        }
//    }
}

ErrorParams::ErrorParams(const toml::value &input) {
//    if (input.contains("Error")) {
//        auto error_input = toml::find(input, "Error");
//        std::string error_file_name_str = "text_outputs/" +
//            toml::find_or<std::string>(error_input, "error_file", "sol_error.txt");
//        strcpy(fname, error_file_name_str.c_str());
//        // std::string res_norm_file_name_str = "text_outputs/" + toml::find_or<std::string>(
//        //     error_input, "residual_norm_file", "residual_norm.txt");
//        // strcpy(res_norm_file_name, res_norm_file_name_str.c_str());
//        interval = toml::find_or<int>(error_input, "error_interval", -1);
//        // res_norm_interval = toml::find_or<int>(error_input, "residual_norm_interval", -1);
//
//        if (interval > 0) {
//            // std::string norm_type_name = toml::find<std::string>(input, "Error", "norm");
//            // if (norm_type_name == "DomainIntegral") {
//            //     norm_type = NormType::DomainIntegral;
//            // }
//            // else if (norm_type_name == "L1") {
//            //     norm_type = NormType::L1;
//            // }
//            // else if (norm_type_name == "L2") {
//            //     norm_type = NormType::L2;
//            // }
//            // else {
//            //     throw InputException("Norm not recognized");
//            // }
//
//            std::vector<std::string> var_names = toml::find<std::vector<std::string>>(
//                input, "Error", "variables");
//            assert(var_names.size() <= N_VAR_OUTPUT_MAX);
//            nvar = var_names.size();
//            for (unsigned ivar = 0; ivar < var_names.size(); ivar++) {
//                strcpy(varsname[ivar], var_names[ivar].c_str());
//            }
//            // get reference solution
//            std::string ref_soln_name = toml::find<std::string>(input, "Error", "ref_soln");
//            analyt_t = get_analytic_type(ref_soln_name);
//        } // if interval is > 0
//        if (error_input.contains("quad_order")) {
//            quad_order = (unsigned) toml::find<int>(error_input, "quad_order");
//            quad_order_same = false;
//        }
//    } // if input file contains an error section
}

IOParams::IOParams(const toml::value &input_info, const std::vector<std::string> &bfg_names) :
    solfileparams(input_info),
    bout(input_info, bfg_names),
    vol_diagnosis(input_info),
    stats(input_info),
    errors(input_info) {}

std::string IOParams::report() const {
//    std::stringstream msg;
//    std::string prefix = "    -> ";
//    msg << "IOParams object reporting:" << "\n";
//    // solution files
//    msg << prefix << "Solution file: \n";
//    msg << prefix << "  write interval                     = " << solfileparams.interval << "\n";
//    msg << prefix << "  #variables                         = " << solfileparams.nvar << "\n";
//    msg << prefix << "  output location                    = " << solfileparams.node_t._to_string()
//        << "\n";
//    msg << prefix << "  output order                       = " << solfileparams.order << "\n";
//    msg << prefix << "  # of post-pro points per elem      = " << solfileparams.nppp << "\n";
//    msg << prefix << "  # of post-pro cells per elem       = " << solfileparams.nppsc << "\n";
//    msg << prefix << "  # of points per post-pro sub-cell  = " << solfileparams.nppscn << "\n";
//    // diagnosis
//    if (vol_diagnosis.interval > 0) {
//        msg << prefix << "Volume diagnosis ON:\n";
//        msg << prefix << "  interval       = " << vol_diagnosis.interval << "\n";
//        msg << prefix << "  # of variables = " << vol_diagnosis.nvar << "\n";
//        if (vol_diagnosis.quad_order_same) {
//            msg << prefix << "  same quadrature order as the solution\n";
//        }
//        else {
//            msg << prefix << "  custom quadrature order of " << vol_diagnosis.quad_order << "\n";
//        }
//    }
//    // boundary diagnosis
//    if (bout.nout > 0) {
//        msg << prefix << "Boundary diagnosis" << "\n"
//            << "  " << prefix << bout.nout << " boundary face groups to look at: ";
//        for (unsigned i = 0; i < bout.nout; i++) {
//            msg << bout.idxs[i] << " ";
//        }
//        msg << "\n";
//    }
//    // statistics
//    if (stats.interval > 0) {
//        msg << prefix << "Statistics ON:\n";
//        msg << prefix << "  interval = " << stats.interval << "\n";
//        if (stats.quad_order_same) {
//            msg << prefix << "  same quadrature order as the solution\n";
//        }
//        else {
//            msg << prefix << "  custom quadrature order of " << stats.quad_order << "\n";
//        }
//    }
//    // errors
//    if (errors.interval > 0) {
//        msg << prefix << "Errors ON:\n";
//        msg << prefix << "  interval = " << errors.interval << "\n";
//        if (errors.quad_order_same) {
//            msg << prefix << "  same quadrature order as the solution\n";
//        }
//        else {
//            msg << prefix << "  custom quadrature order of " << errors.quad_order << "\n";
//        }
//    }
//    return msg.str();
}
