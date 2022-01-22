#include <type_traits>
#include "common/my_exceptions.h"
#include "io/input_info.h"
//#include "equations/analytic_expression_default.h"
#include "mesh/mesh.h"

static_assert(std::is_trivially_copyable<InputInfo>(), "SchemeInfo is not trivially copyable");

InputInfo::InputInfo(const toml::value &toml_input, const Mesh &mesh) :
    //solver_params(toml_input),
    num_params(toml_input, mesh.order),
    //step_params(toml_input),
    io_params(toml_input, mesh.BFGnames) {
//
//    /* This branch will take advantage of diagonal mass matrices.
//     * The current layout is not suited for element-wise operations that couple only in nb.
//     * This is a problem that I need to think about but for the moment, restrict the implementation.
//     * */
//    {
//        const BasisType type = num_params.basis;
//        assert(
//            (type._to_index() == BasisType::Legendre2D) ||
//            (type._to_index() == BasisType::Legendre3D) ||
//            (type._to_index() == BasisType::LagrangeGL2D) ||
//            (type._to_index() == BasisType::LagrangeGL3D)
//        );
//    }
//
//    std::string name;
//
//    /* Depending on the requested physics type, we initialize only the right
//     * physics data object. */
//    auto physics_info = toml::find(toml_input, "Physics");
//    name = toml::find_or<std::string>(physics_info, "name", "IdealGas");
//    dim = toml::find<unsigned>(physics_info, "dim");
//    physics_type = PhysicsType::_from_string(name.c_str());
//    switch (physics_type) {
//        case PhysicsType::IdealGas: {
//            ig_params = Physics::IdealGasParams(toml_input);
//            ns = dim + 2;
//            break;
//        }
//        case PhysicsType::MultiSpecies: {
//            ms_params = Physics::MultiSpeciesParams(toml_input);
//            ns = dim + 2 + ms_params.nsp - 1;
//            break;
//        }
//        default: {
//            throw InputException("Unrecognized physics type " + name);
//        }
//    }
//
//    // stiff integrator for chemistry
//    if (physics_type._to_index() == PhysicsType::MultiSpecies) {
//        rplus_params = Math::ROWPlusParams(toml_input);
//        direct_integration = toml::find_or<bool>(toml_input, "direct_integration", false);
//        integration_t = toml::find_or<int>(toml_input, "integration_type", 0);
//    }
//
//    nelem = mesh.nelem;
//    nnode_per_elem = mesh.nnode_per_elem;
//    ncoeff = num_params.nb * ns;
//    ncoord = nnode_per_elem * dim;
//    nface_per_elem = mesh.nIF_in_elem[0]; // TODO Make this more robust
//    assert(nface_per_elem == 4 || nface_per_elem == 6 || nface_per_elem == 3);
//    assert(ns <= N_STATE_MAX);
//    assert(ncoeff <= N_COEFFS_MAX);
//
//    /* Since in NumericsParams, the geometric basis type is solely determined by
//     * the solution basis type provided in the input file, add a temporary check here
//     * to make sure that the number of geometric basis is consistent with the mesh file. */
//    if (num_params.gnb != mesh.nnode_per_elem) {
//        std::stringstream ss;
//        ss << "The number of geometric basis function is not consisent with the mesh file:"
//           << " num_params.gnb = " << num_params.gnb
//           << " mesh.nnode_per_elem = " << mesh.nnode_per_elem;
//        throw InputException(ss.str());
//    }
//
//    /* The initialization of the EquationData object is left here because of the introduction
//     * of input data for initialization by Steven. Please move this away in a consistent
//     * way compared to other Data constructors.
//     * TODO move this */
//
//    // general
//    auto eq_info = toml::find(toml_input, "Equation");
//    std::string eq_name = toml::find<std::string>(eq_info, "name");
//    std::string inv_flux_name = toml::find_or<std::string>(eq_info, "inviscid_flux", "HLLC");
//    std::string visc_flux_name = toml::find_or<std::string>(eq_info, "viscous_flux", "IP");
//    eq_params.type = get_equation_type(eq_name, inv_flux_name, visc_flux_name);
//    if (eq_params.type==EquationType::None) {
//        throw InputException("Equation not recognized in SchemeInfo.");
//    }
//    // initialization
//    name = toml::find<std::string>(eq_info, "init_type");
//    eq_params.init_t = get_analytic_type(name);
//    if (eq_params.init_t==AnalyticType::None) {
//        throw InputException("Unrecognized initialization type");
//    }
//    std::vector<rtype> default_init_data = default_analytic_params(dim, eq_params.init_t);
//    if (toml_input.at("Equation").contains("init_data")) {
//        std::vector<rtype> init_data=toml::find<std::vector<rtype>>(eq_info, "init_data");
//        assert(init_data.size()<=INIT_EX_PARAMS_MAX);
//        if (init_data.size()!=default_init_data.size()) {
//            std::string err_msg = "Init data needs " + std::to_string(default_init_data.size())
//                + " params. Has " + std::to_string(init_data.size())+ ".";
//            throw InputException(err_msg);
//        }
//        memcpy(eq_params.init_ex_params, init_data.data(), init_data.size()*sizeof(rtype));
//    }
//    else {
//        memcpy(eq_params.init_ex_params,default_init_data.data(),
//            default_init_data.size()*sizeof(rtype));
//    }
//    // source
//    name = toml::find_or<std::string>(eq_info, "source", "None");
//    eq_params.src_t = get_source_type(name); // OK to be none
//    // boundary information
//    Equation::BoundaryData *bdata = &(eq_params.bdata[0]);
//    eq_params.nBFG = mesh.nBFG;
//    assert(eq_params.nBFG <= N_BFG_MAX);
//    int iBFG = 0;
//    for (std::string bname: mesh.BFGnames) {
//        std::string btype_name = toml::find<std::string>(
//            toml_input, "Boundaries", bname, "type");
//        bdata[iBFG].type = get_boundary_type(btype_name);
//        if (bdata[iBFG].type==BoundaryType::None) {
//            std::stringstream msg;
//            msg << "Boundary type for " << bname << " boundary not recognized. Got "
//                << btype_name;
//            throw InputException(msg.str());
//        }
//        process_bdata(iBFG, bname, toml_input);
//        iBFG++;
//    }
//
//    /*======================================*/
//    /*-------------  OUTPUTS  --------------*/
//    /*======================================*/
//
//    /* What comes below has not been considered for refactoring yet. */
//
//    // restart
//    restart_interval = toml::find_or<int>(toml_input, "restart_interval", -1);
//    std::string prefix = "restarts/restart"; // TODO unhardcode this
//    prefix.copy(restart_file_prefix, prefix.size()+1);
//    restart_file_prefix[prefix.size()] = '\0';
//    if (toml_input.at("Stepper").contains("restart")) {
//        use_restart = true;
//        std::string fname = toml::find<std::string>(toml_input, "Stepper", "restart");
//        strcpy(restart_file_name, fname.c_str());
//    }
//    else {
//        use_restart = false;
//    }
}

std::string InputInfo::report() const {
//    std::stringstream msg;
//    msg << std::string(80, '=') << std::endl;
//
//    // report this structure
//    msg << "InputInfo object reporting: " << std::endl << std::endl;
//    msg << "dim              = " << dim << std::endl
//        << "ns               = " << ns << std::endl
//        << "ncoeff           = " << ncoeff << std::endl
//        << "nelem            = " << nelem << std::endl
//        << "nnode_per_elem   = " << nnode_per_elem << std::endl
//        << "nface_per_elem   = " << nface_per_elem << std::endl
//        << "restart_interval = " << restart_interval << std::endl;
//
//    // for multi-species physics, report some macros
//    if (physics_type._to_index() == PhysicsType::MultiSpecies) {
//        msg << "\n"
//            << "N_SPECIES        = " << N_SPECIES << std::endl
//            << "N_REACTIONS      = " << N_REACTIONS << std::endl;
//#ifdef MIX1
//        msg << "Transport        = MIX1" << std::endl;
//#else
//        msg << "Transport        = MIX2" << std::endl;
//#endif
//#ifdef USE_MS_CLIPPING
//        msg << "USE_MS_CLIPPING  = ON" << std::endl;
//#else
//        msg << "USE_MS_CLIPPING  = OFF" << std::endl;
//#endif
//        if (direct_integration) {
//            msg << "USING DIRECT INTEGRATION!!!!!!!!!" << std::endl;
//        }
//        msg << "integration type = " << integration_t << std::endl;
//    }
//
//    if (use_restart) {
//        msg << "\n" << "Using restart: " << restart_file_name << std::endl;
//    }
//
//    msg << "\n" << solver_params.report();
//    msg << "\n" << num_params.report();
//
//    if (physics_type._to_index() == PhysicsType::IdealGas) {
//        msg << "\n" << ig_params.report();
//    }
//    else if (physics_type._to_index() == PhysicsType::MultiSpecies) {
//        msg << "\n" << ms_params.report();
//    }
//
//    if (ms_params.reacting) {
//        msg << "\n" << rplus_params.report();
//    }
//
//    msg << "\n" << io_params.report();
//
//    if (eq_params.nBFG>0) {
//        msg << "\n" << eq_params.nBFG << " boundaries:" << std::endl;
//        for (int i=0; i<eq_params.nBFG; i++) {
//            msg << "Boundary " << i << " of type "
//                << (int) eq_params.bdata[i].type << std::endl;
//            if (eq_params.bdata[i].has_data) {
//                msg << "Data: ";
//                for (uint j=0; j<eq_params.bdata[i].nData; j++) {
//                    msg << eq_params.bdata[i].data[j] << " ";
//                }
//                msg << std::endl;
//            }
//        }
//    }
//
//    msg << std::string(80, '=') << std::endl << std::endl;
//    return msg.str();
}

void InputInfo::process_bdata(
    const int iBFG,
    const std::string &bname,
    const toml::value &toml_input) {

//    Equation::BoundaryData *bdata = &(eq_params.bdata[0]);
//    if (toml_input.at("Boundaries").at(bname).contains("data")) {
//        bdata[iBFG].has_data = true;
//        auto data_vec = toml::find<std::vector<rtype>>(toml_input, "Boundaries", bname, "data");
//        assert(data_vec.size() < N_BDATA_MAX);
//        bdata[iBFG].nData = data_vec.size();
//        std::copy_n(data_vec.data(), data_vec.size(), bdata[iBFG].data);
//    }
//    if (toml_input.at("Boundaries").at(bname).contains("analytic_type")) {
//        std::string analytic_type_name = toml::find<std::string>(toml_input, "Boundaries",
//            bname, "analytic_type");
//        bdata[iBFG].analytic_type = get_analytic_type(analytic_type_name);
//    }
}
