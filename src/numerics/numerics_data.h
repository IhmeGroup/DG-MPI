#ifndef DG_NUMERICS_DATA_H
#define DG_NUMERICS_DATA_H

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include "toml11/toml.hpp"
#include "common/defines.h"
#include "common/enums.h"

using std::string, std::vector;


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
