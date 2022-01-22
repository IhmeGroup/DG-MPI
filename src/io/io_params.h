#ifndef DG_IO_PARAMS_H
#define DG_IO_PARAMS_H

#include <string>
#include <vector>
#include "toml11/toml.hpp"
#include "common/defines.h"
#include "numerics/numerics_data.h"

enum class PhysicalVariable;
enum class AnalyticType;

/*! \brief Struct to hold input information about boundary diagnosis
 *
 */
struct BoundaryOutputParams {
  public:
    BoundaryOutputParams(const toml::value &toml_input, const std::vector<std::string> &bfg_names);

  public:
    unsigned nout = 0; //!< number of boundary face groups for which diagnosis is requested
    unsigned idxs[N_BFG_MAX]; //!< array of boundary face group indexes for which diagnosis is requested
    unsigned nvar[N_BFG_MAX]; //!< array of the number of variables for each boundary face group index in #idxs
    char var_names[N_BFG_MAX][N_VAR_BDIAG_MAX][VAR_NAME_LEN_MAX]; //!< contains the variable name for each BFG, for each variable
    char boundary_names[N_BFG_MAX][VAR_NAME_LEN_MAX]; //!< contains the name for each BFG
};

struct SolutionFileParams {
  public:
    SolutionFileParams(const toml::value &input);

  public:
    NodeType node_t;
    unsigned order;
    unsigned nppp; //!< number of post-processing points per element
    unsigned nppsc; //!< number of post-processing sub-cells per element
    unsigned nppscn; //!< number of nodes per-post-processing sub-cell
    int interval;
    char prefix[FILE_NAME_LEN_MAX]; //!< prefix of the solution files' name
    unsigned nvar; //!< number of variables to output in the solution file
    PhysicalVariable vars[N_VAR_OUTPUT_MAX]; //!< list of output variables
    char varsname[N_VAR_OUTPUT_MAX][VAR_NAME_LEN_MAX]; //!< list of output variables name
};

struct VolumeDiagnosisParams {
  public:
    VolumeDiagnosisParams(const toml::value &input);

  public:
    int interval = -1;
    unsigned quad_order = 0;
    bool quad_order_same = true;
    unsigned nvar = 0;
    char varsname[N_VAR_OUTPUT_MAX][VAR_NAME_LEN_MAX]; //!< list of output variables name
};

struct StatsParams {
  public:
    StatsParams(const toml::value &input);

  public:
    int interval = -1;
    unsigned quad_order = 0;
    bool quad_order_same = true;
};

struct ErrorParams {
  public:
    ErrorParams(const toml::value &input);

  public:
    char fname[FILE_NAME_LEN_MAX];
    int interval = -1;
    unsigned quad_order = 0;
    bool quad_order_same = true;
    int nvar = 0;
    char varsname[N_VAR_OUTPUT_MAX][VAR_NAME_LEN_MAX];
    AnalyticType analyt_t;
};

/*! \brief Wrapper struct to hold input information about IO
 *
 */
struct IOParams {
  public:
    IOParams(const toml::value &input_info, const std::vector<std::string> &bfg_names);
    std::string report() const;

  public:
    SolutionFileParams solfileparams;
    BoundaryOutputParams bout;
    VolumeDiagnosisParams vol_diagnosis;
    StatsParams stats;
    ErrorParams errors;
};

#endif //DG_IO_PARAMS_H
