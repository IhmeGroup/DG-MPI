#ifndef DG_NUMERICS_DATA_H
#define DG_NUMERICS_DATA_H

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include "toml11/toml.hpp"
#include "common/defines.h"

using std::string, std::vector;

// This template returns the string-valued name of an enum value. The template
// parameter should be set to the enum type.
// Example:
//     enum_to_string<BasisType>("Legendre2D")
template <class T>
string enum_to_string(T enum_value);

// This template returns the enum from its string-valued name. The template
// parameter should be set to the enum type.
// Example:
//     enum_from_string<BasisType>(BasisType::Legendre2D)
template <class T>
T enum_from_string(string str);

// This template is used to point towards the correct array containing the enum
// string names.
template <class T>
vector<string> get_enum_to_string_array();

enum BasisType {
    LagrangeEq1D,
    LagrangeEq2D,
    LagrangeEq3D,
    LagrangeGLL1D,
    LagrangeGLL2D,
    LagrangeGLL3D,
    LagrangeGL2D,
    LagrangeGL3D,
    Legendre2D,
    Legendre3D,
    TriangleLagrange
};
const vector<string> basis_type_to_string_array = {
    "LagrangeEq1D",
    "LagrangeEq2D",
    "LagrangeEq3D",
    "LagrangeGLL1D",
    "LagrangeGLL2D",
    "LagrangeGLL3D",
    "LagrangeGL2D",
    "LagrangeGL3D",
    "Legendre2D",
    "Legendre3D",
    "TriangleLagrange"
};

enum QuadratureType {
    GaussLegendre,
    GaussLobatto,
    TriangleDunavant,
    TriangleGaussLegendre
};
const vector<string> quadrature_type_to_string_array = {
    "GaussLegendre",
    "GaussLobatto",
    "TriangleDunavant",
    "TriangleGaussLegendre"
};

enum NodeType {
    Equidistant, // equidistant nodes in reference space
    Interior // equidistant nodes but in the interior of the reference element, only useful for uniform quad/hex meshes
};
const vector<string> node_type_to_string_array = {
    "Equidistant",
    "Interior"
};

enum NormType {
    DomainIntegral,
    L1,
    L2
};
const vector<string> norm_type_to_string_array = {
    "DomainIntegral",
    "L1",
    "L2"
};

enum ShapeType {
    Line,
    Quadrilateral,
    Hexahedron,
    Triangle
};
const vector<string> shape_type_to_string_array = {
    "Line",
    "Quadrilateral",
    "Hexahedron",
    "Triangle"
};

enum DGSchemeType {
    DG,
    SumFact,
    LegendreStruct,
    ColocatedGL,
    ColocatedGLL,
    None
};
const vector<string> dgscheme_type_to_string_array = {
    "DG",
    "SumFact",
    "LegendreStruct",
    "ColocatedGL",
    "ColocatedGLL",
    "None"
};

enum InitType {
    L2Projection,
    Interpolation
};
const vector<string> init_type_to_string_array = {
    "L2Projection",
    "Interpolation"
};

namespace EnumMap {

unsigned get_num_bases(const BasisType type, const unsigned order);
unsigned get_num_bases_1d(const BasisType type, const unsigned order);
unsigned get_num_quad(const QuadratureType type, const unsigned dim, const unsigned order);
unsigned get_num_quad_1d(const QuadratureType type, const unsigned order);
unsigned get_iMM_size(const DGSchemeType type, const unsigned nb);
void get_solfile_params(
    const BasisType type,
    const unsigned order,
    unsigned &nppp,
    unsigned &nppsc,
    unsigned &nppscn);

} // namespace EnumMap

namespace Numerics {

/*! \brief Structure to hold input data related to the numerics
 *
 */
struct NumericsParams {
  public:
    /*! \brief Default constructor
     *
     */
    NumericsParams() = default;
    /*! \brief Constructor
     *
     * @param input_info TOML input file object
     * @param _gorder order of the iso-parametric mapping
     */
    NumericsParams(const toml::value &input_info, const unsigned gorder_);
    /*! \brief Report the content of the object
     *
     * @return
     */
    std::string report() const;

  public:
    BasisType basis; //!< basis type for the solution
    BasisType gbasis; //!< basis type for the isoparametric mapping
    QuadratureType quad; //!< quadrature type
    DGSchemeType scheme; //!< DG scheme type
    InitType init = InitType::L2Projection; //!< type of numerical initialization
  public:
    unsigned order; //!< order for the solution
    unsigned gorder; //!< order for the isoparametric mapping
    unsigned quad_order; //!< order for the quadrature (the order up to which the rule will be exact)
    unsigned nb; //!< number of basis functions for the solution
    unsigned nb1d; //!< number of 1D basis functions, for tensor-product bases with sum-factorization optimization
    unsigned gnb; //!< number of basis functions for the geometric mapping
    unsigned nq; //!< number of volume quadruature points
    unsigned nq1d; //!< number of 1D quadrature points, for tensor-product bases with sum-factorization optimization
    unsigned nqf; //!< number of face quadrature points
    unsigned nbact; //!< actual relevant number of basis functions, depends on whether or not sum-factorization is on, in which case it is set to nb1d
    unsigned nqact; //!< similar to nbact
    unsigned nqfact; //!< similar to nbact
  public:
    unsigned nface; //!< number of faces per element
    unsigned norient; //!< number of face orientations per face
    bool over_integrate_MM = false; //!< whether to over-integrate the mass matrix
};

} // namespace Numerics

#endif //DG_NUMERICS_DATA_H
