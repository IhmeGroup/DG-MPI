#ifndef DG_DEFINES_H
#define DG_DEFINES_H

// compiler directive to switch between single and double precision
#ifdef USE_DOUBLES
using rtype = double;
#else
using rtype = float;
#endif

#define INIT_EX_PARAMS_MAX 10

#define N_BASIS_MAX 300  //!< maximum number of basis function
#define N_STATE_MAX 20 //!< maximum number of state variables
#define N_COEFFS_MAX (N_BASIS_MAX * N_STATE_MAX) //!< maximum number of solution coefficients (ns*nb)
#define N_FACE_MAX 8 //!< maximum number of faces (8 = hexahedron)
#define N_BFG_MAX 5 //!< maximum number of boundary face groups
#define N_BDATA_MAX 10 //!< maximum number of boundary data per boundary face group
#define N_VAR_OUTPUT_MAX 40 //!< maximum number of variables for output, diagnosis etc.
#define N_VAR_BDIAG_MAX 10 //!< maximum number of variables for boundary diagnosis, less than N_VAR_OUTPUT_MAX because of Legion limit on return type's size

#define FILE_NAME_LEN_MAX 128 //!< maximum length of a file name
#define VAR_NAME_LEN_MAX 128 //!< maximum length of a variable name

// #ifdef DG_USE_KOKKOS
/* It is the responsibility of the file using Kokkos to include the Kokkos headers.
 * Namely, include <Kokkos_Core.hpp> guarded by DG_USE_KOKKOS
 * for the following macros to be defined. */
#define DG_KOKKOS_FUNCTION KOKKOS_FUNCTION
#define DG_KOKKOS_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
#define DG_KOKKOS_FORCEINLINE_FUNCTION KOKKOS_FORCEINLINE_FUNCTION
#else
/* If Kokkos is not used, then these macros are empty which ensures backward compatibility. */
// #define DG_KOKKOS_FUNCTION
// #define DG_KOKKOS_INLINE_FUNCTION inline
// #define DG_KOKKOS_FORCEINLINE_FUNCTION inline
// #endif

#endif //DG_DEFINES_H
