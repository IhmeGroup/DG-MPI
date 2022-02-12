#include <cmath>
#include <iostream>
#include "common/my_exceptions.h"
#include "numerics/numerics_data.h"

using namespace std;


namespace EnumMap {

unsigned get_num_bases(const BasisType type, const unsigned order) {
    switch (type) {
        case BasisType::LagrangeSeg:
        case BasisType::LagrangeGLL1D:
            return order + 1;
            break;
        case BasisType::LagrangeQuad:
        case BasisType::LagrangeGLL2D:
        case BasisType::LagrangeGL2D:
        case BasisType::LegendreQuad:
            return (order + 1) * (order + 1);
            break;
        case BasisType::LagrangeHex:
        case BasisType::LagrangeGLL3D:
        case BasisType::LagrangeGL3D:
        case BasisType::LegendreHex:
            return (order + 1) * (order + 1) * (order + 1);
            break;
        case BasisType::TriangleLagrange:
            return (order + 1) * (order + 2) / 2;
            break;
        default: {
            throw InputException("Basis not recognized in get_num_bases");
        }
    }
}

unsigned get_num_bases_1d(const BasisType type, const unsigned order) {
    switch (type) {
        case BasisType::LagrangeSeg:
        case BasisType::LagrangeGLL1D:
        case BasisType::LagrangeQuad:
        case BasisType::LagrangeGLL2D:
        case BasisType::LagrangeGL2D:
        case BasisType::LegendreQuad:
        case BasisType::LagrangeHex:
        case BasisType::LagrangeGLL3D:
        case BasisType::LagrangeGL3D:
        case BasisType::LegendreHex:
            return order + 1;
        case BasisType::TriangleLagrange:
            return 0.;
        default: {
            throw InputException("Basis not recognized in get_num_bases");
        }
    }
}

unsigned get_num_quad(const QuadratureType type, const unsigned dim, const unsigned order) {
    int nq = 1;
    switch (type) {
        case QuadratureType::GaussLegendre: {
            return pow(ceil((order + 1.) / 2.), dim);
        }
        case QuadratureType::GaussLobatto: {
            return pow(1 + ceil((order + 1.) / 2.), dim);
        }
        case QuadratureType::TriangleGaussLegendre: {
            return pow(ceil((order + 2.) / 2.), dim);
        }
        case QuadratureType::TriangleDunavant: {
            if (dim == 1) {
                return pow(ceil((order + 1.) / 2.), dim);
            }
            switch (order) {
                case 0 :
                    nq = 1;
                    break;
                case 1 :
                    nq = 1;
                    break;
                case 2 :
                    nq = 3;
                    break;
                case 3 :
                    nq = 4;
                    break;
                case 4 :
                    nq = 6;
                    break;
                case 5 :
                    nq = 7;
                    break;
                case 6 :
                    nq = 12;
                    break;
                case 7 :
                    nq = 13;
                    break;
                case 8 :
                    nq = 16;
                    break;
                case 9 :
                    nq = 19;
                    break;
                case 10 :
                    nq = 25;
                    break;
                case 11 :
                    nq = 27;
                    break;
                case 12 :
                    nq = 33;
                    break;
                case 13 :
                    nq = 37;
                    break;
                case 14 :
                    nq = 42;
                    break;
                case 15 :
                    nq = 48;
                    break;
                case 16 :
                    nq = 52;
                    break;
                case 17 :
                    nq = 61;
                    break;
                case 18 :
                    nq = 70;
                    break;
                case 19 :
                    nq = 73;
                    break;
                case 20 :
                    nq = 79;
                    break;
                default:
                    break;
            }
            return nq;
        }
        default: {
            throw InputException("Quadrature not recognized in get_num_quad");
        }
    }
}

unsigned get_num_quad_1d(const QuadratureType type, const unsigned order) {
    switch (type) {
        case QuadratureType::GaussLegendre: {
            return ceil((order + 1.) / 2.);
        }
        case QuadratureType::GaussLobatto: {
            return 1 + ceil((order + 1.) / 2.);
        }
        case QuadratureType::TriangleGaussLegendre: {
            return 0;
        }
        case QuadratureType::TriangleDunavant: {
            return 0;
        }
        default: {
            throw InputException("Quadrature not recognized in get_num_quad");
        }
    }
}

unsigned get_iMM_size(const DGSchemeType type, const unsigned nb) {
    switch (type) {
        case DGSchemeType::LegendreStruct: {
            return nb;
        }
        default: {
            return nb * nb;
        }
    }
}

void get_solfile_params(
    const BasisType type,
    const unsigned order,
    unsigned &nppp,
    unsigned &nppsc,
    unsigned &nppscn) {
            printf("CHFKD:A");

    switch (type) {
        case BasisType::LagrangeQuad :
        case BasisType::LagrangeGLL2D :
        case BasisType::LagrangeGL2D :
        case BasisType::LegendreQuad : {
            nppp = pow(std::max(order, 1u) + 1, 2);
            nppsc = pow(std::max(order, 1u), 2);
            nppscn = 4;
            break;
        }

        case BasisType::LagrangeHex :
        case BasisType::LagrangeGLL3D :
        case BasisType::LagrangeGL3D :
        case BasisType::LegendreHex : {
            nppp = pow(std::max(order, 1u) + 1, 3);
            nppsc = pow(std::max(order, 1u), 3);
            nppscn = 8;
            break;
        }

        case BasisType::TriangleLagrange : {
            nppp = (std::max(order, 1u) + 1)*(std::max(order, 1u) + 2)/2;
            nppsc = (std::max(order, 1u))*(std::max(order, 1u));
            nppscn = 3;
            break;
        }

        default: {
            std::stringstream msg;
            msg << "Unsupported basis type " << enum_to_string<BasisType>(type) << " in " << __PRETTY_FUNCTION__;
            throw FatalException(msg.str());
        }
    }
}

} // namespace EnumMap

namespace Numerics {

NumericsParams::NumericsParams(const toml::value &input_info, const unsigned gorder_) {
    const unsigned dim = (unsigned) toml::find<int>(input_info, "Physics", "dim");
    const auto num_info = toml::find(input_info, "Numerics");

    // solution basis
    order = (unsigned) toml::find<int>(num_info, "order");
    string name = toml::find<std::string>(num_info, "basis");
    basis = enum_from_string<BasisType>(name.c_str());
    nb = EnumMap::get_num_bases(basis, order);
    nb1d = EnumMap::get_num_bases_1d(basis, order);
    assert(nb <= N_BASIS_MAX);

    // geometric basis
    // currently, the right geometric basis is determined from the solution basis
    gorder = gorder_;
    switch (basis) {
        case BasisType::LagrangeSeg:
        case BasisType::LagrangeGLL1D: {
            gbasis = BasisType::LagrangeSeg;
            nface = 2;
            norient = 1;
            break;
        }
        case BasisType::LagrangeQuad:
        case BasisType::LagrangeGLL2D:
        case BasisType::LagrangeGL2D:
        case BasisType::LegendreQuad: {
            gbasis = BasisType::LagrangeQuad;
            nface = 4;
            norient = 2;
            break;
        }
        case BasisType::LagrangeHex:
        case BasisType::LagrangeGLL3D:
        case BasisType::LagrangeGL3D:
        case BasisType::LegendreHex: {
            gbasis = BasisType::LagrangeHex;
            nface = 6;
            norient = 8;
            break;
        }
        case BasisType::TriangleLagrange: {
            gbasis = BasisType::TriangleLagrange;
            nface = 3;
            norient = 2;
            break;
        }
        default:
            throw InputException("Solution basis has no matching geometric basis");
            break;
    }
    gnb = EnumMap::get_num_bases(gbasis, gorder);

    // quadrature
    string quad_name;
    switch (basis) {
        case BasisType::LagrangeSeg:
        case BasisType::LagrangeGLL1D:
        case BasisType::LagrangeQuad:
        case BasisType::LagrangeGLL2D:
        case BasisType::LagrangeGL2D:
        case BasisType::LegendreQuad:
        case BasisType::LagrangeHex:
        case BasisType::LagrangeGLL3D:
        case BasisType::LagrangeGL3D:
        case BasisType::LegendreHex: {
            quad_name = toml::find_or<string>(num_info, "quad", "GaussLegendre");
            break;
        }
        case BasisType::TriangleLagrange: {
            quad_name = toml::find_or<string>(num_info, "quad", "TriangleDunavant");
            break;
        }
        default:
            throw InputException("Solution basis has no matching default quadrature rule");
            break;
    }

    quad = enum_from_string<QuadratureType>(quad_name.c_str());
//    if (
//        (
//            (gbasis._to_index() == BasisType::TriangleLagrange) &&
//            ( (quad._to_index() != QuadratureType::TriangleDunavant) &&
//              (quad._to_index() != QuadratureType::TriangleGaussLegendre) )
//        ) ||
//        (
//            (gbasis._to_index() != BasisType::TriangleLagrange) &&
//            ( (quad._to_index() == QuadratureType::TriangleDunavant) ||
//              (quad._to_index() == QuadratureType::TriangleGaussLegendre) )
//        )
//       ) {
//        throw InputException("Mismatch shape and quad rule");
//    }
//
//    /* The default value is for general quad/hex knowing that the Jacobian's determinant
//     * is constant only for perfect squares/cubes. */
//    quad_order = toml::find_or<int>(num_info, "quad_order", 2 * (order + gorder) + 1);
//
//    // DG Scheme: this can overwrite the quadrature rule
//    string scheme_name = toml::find_or<string>(num_info, "scheme", "DG");
//    scheme = DGSchemeType::_from_string(scheme_name.c_str());
//    if (scheme._to_index() == DGSchemeType::ColocatedGL) {
//        if (basis._to_index() != BasisType::LagrangeGL2D &&
//            basis._to_index() != BasisType::LagrangeGL3D) {
//            throw InputException("ColocatedGL is only compatible with a GL Lagrange basis.");
//        }
//        // readjust quadrature type and order for colocation
//        quad = QuadratureType::GaussLegendre;
//        quad_order = 2 * order + 1;
//    }
//    if (scheme._to_index() == DGSchemeType::ColocatedGLL) {
//        if (basis._to_index() != BasisType::LagrangeGLL2D &&
//            basis._to_index() != BasisType::LagrangeGLL3D) {
//            throw InputException("ColocatedGLL is only compatible with a GLL Lagrange basis.");
//        }
//        // readjust quadrature type and order for colocation
//        quad = QuadratureType::GaussLobatto;
//        quad_order = 2 * order - 1;
//    }
//
//    // now that we know the scheme, we can compute the number of quadrature points
//    nq = EnumMap::get_num_quad(quad, dim, quad_order);
//    nq1d = EnumMap::get_num_quad_1d(quad, quad_order);
//    nqf = EnumMap::get_num_quad(quad, dim-1, quad_order);
//
//    // initialization type
//    init = InitType::_from_string(toml::find_or<string>(num_info, "init", "L2Projection").c_str());
//    if (init._to_index() == InitType::Interpolation) {
//        if (basis._to_index() != BasisType::LagrangeEq1D &&
//            basis._to_index() != BasisType::LagrangeQuad &&
//            basis._to_index() != BasisType::LagrangeEq3D &&
//            basis._to_index() != BasisType::LagrangeGL2D &&
//            basis._to_index() != BasisType::LagrangeGL3D &&
//            basis._to_index() != BasisType::LagrangeGLL1D &&
//            basis._to_index() != BasisType::LagrangeGLL2D &&
//            basis._to_index() != BasisType::LagrangeGLL3D &&
//            basis._to_index() != BasisType::TriangleLagrange
//            ) {
//            throw InputException("Interpolation initialization is only compatible with a "
//                                 "Lagrange basis.");
//        }
//    }
//
//    // quadrature order for the mass matrix
//    over_integrate_MM = toml::find_or<bool>(num_info, "over_integrate_MM", false);
//    if (over_integrate_MM) {
//        throw NotImplementedException("Over-integration of MM not implemented");
//    }
//
//    // nbact etc. will be used to transparently allow the use of sum-factorization optimization
//    if (scheme._to_index() == DGSchemeType::SumFact) {
//        nbact = nb1d;
//        nqact = nq1d;
//        nqfact = nq1d;
//    }
//    else {
//        nbact = nb;
//        nqact = nq;
//        nqfact = nqf;
//    }
}

string NumericsParams::report() const {
    stringstream msg;
    string prefix = "    -> ";
    msg << "NumericsData object reporting:" << endl;
//    msg << prefix << "Basis type                        = " << basis._to_string() << endl
//        << prefix << "Geometric basis type              = " << gbasis._to_string() << endl
//        << prefix << "Quadrature type                   = " << quad._to_string() << endl
//        << prefix << "DG scheme type                    = " << scheme._to_string() << endl
//        << prefix << "Initialization                    = " << init._to_string() << endl
//        << prefix << "Order                             = " << order << endl
//        << prefix << "Geometric order                   = " << gorder << endl
//        << prefix << "Quadrature order                  = " << quad_order << endl
//        << prefix << "# of solution bases               = " << nb << endl
//        << prefix << "# of geometric bases              = " << gnb << endl
//        << prefix << "# of volume quadrature points     = " << nq << endl
//        << prefix << "# of face quadrature points       = " << nqf << endl
//        << prefix << "# of faces per element            = " << nface << endl
//        << prefix << "# of orientations per face        = " << norient << endl;
//    if (over_integrate_MM) {
//        msg << prefix << "Over-integrating the mass matrix" << endl;
//    }
    return msg.str();
}

} // namespace Numerics
