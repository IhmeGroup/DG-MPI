#ifndef DG_INPUT_INFO_H
#define DG_INPUT_INFO_H

#include <string>
#include "toml11/toml.hpp"
#include "common/defines.h"

//#include "exec/solver_params.h"
//#include "physics/physics_data.h"
//#include "equations/equation_data.h"
//#include "math/math_data.h"
#include "numerics/numerics_data.h"
//#include "stepper/stepper_data.h"
#include "io/io_params.h"

class Mesh;

/*! \brief InputInfo structure
 *
 * This structure directly translates the information contained in the input file to a custom
 * structure. This structure needs to be trivially copyable without pointers in order to be
 * moved around by Legion. Once the input file has been processed using the TOML library into
 * this struct, the toml object should be discarded.
 */
struct InputInfo {
  public:
    /*! \brief InputInfo constructor
     *
     * @param toml_input TOML input file object
     * @param mesh Mesh object
     */
    InputInfo(const toml::value &toml_input, const Mesh &mesh);

    /*! \brief Report the content of the object
     *
     * @return
     */
    std::string report() const;

  private:
    /*! \brief Process one boundary face group
     *
     * @param iBFG index of the boundary face group
     * @param bname boundary face group name
     * @param toml_input TOML input file object
     */
    void process_bdata(const int iBFG, const std::string &bname, const toml::value &toml_input);

  public:
    //SolverParams solver_params; //!< solver's parameters
    Numerics::NumericsParams num_params; //!< numerics' parameters
    //Stepper::StepperParams step_params; //!< stepper's parameters
    IOParams io_params; //!< IO's parameters
    //PhysicsType physics_type; //!< type of physics
    /* One of the two physics data is invalid. */
    //Physics::IdealGasParams ig_params; //!< ideal gas physics parameters
    //Physics::MultiSpeciesParams ms_params; //!< multi-species physics parameters
    //Equation::EquationParams eq_params; //!< equation parameters
    //Math::ROWPlusParams rplus_params; //!< ROWPlus ODE integrator parameters
  public:
    unsigned dim; //!< number of spatial dimensions
    unsigned ns; //!< number of state variables
    unsigned ncoeff; //!< number of polynomial coefficients per element
    unsigned nelem; //<! number of elements in the mesh
    unsigned nnode_per_elem; //!< number of mesh nodes per element
    unsigned ncoord; //!< number of coordinate data per element
    unsigned nface_per_elem; //!< number of faces per element
    bool direct_integration;
    unsigned integration_t = 0;

    /* What is below has not been considered for refactoring. */

    // restart files
    bool use_restart; //!< use restart file
    int restart_interval; //!< interval at which to write restart
    char restart_file_prefix[FILE_NAME_LEN_MAX];
    char restart_file_name[FILE_NAME_LEN_MAX]; //!< restart file
};

#endif //DG_INPUT_INFO_H
