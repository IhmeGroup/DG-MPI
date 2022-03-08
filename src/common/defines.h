#ifndef DG_DEFINES_H
#define DG_DEFINES_H

#include <Kokkos_Core.hpp>
#include <mpi.h>

// compiler directive to switch between single and double precision
#ifdef USE_DOUBLES
using rtype = double;
constexpr auto MPI_RTYPE = MPI_DOUBLE;
#else
using rtype = float;
constexpr auto MPI_RTYPE = MPI_FLOAT;
#endif

constexpr auto INIT_EX_PARAMS_MAX = 10;
constexpr auto INIT_NAME_MAX_CHARS = 30;

constexpr auto N_BASIS_MAX = 300;  //!< maximum number of basis function
constexpr auto N_STATE_MAX = 20; //!< maximum number of state variables
constexpr auto N_COEFFS_MAX = N_BASIS_MAX * N_STATE_MAX; //!< maximum number of solution coefficients (ns*nb)
constexpr auto N_FACE_MAX = 8; //!< maximum number of faces (8 = hexahedron)
constexpr auto N_BFG_MAX = 5; //!< maximum number of boundary face groups
constexpr auto N_BDATA_MAX = 10; //!< maximum number of boundary data per boundary face group
constexpr auto N_VAR_OUTPUT_MAX = 40; //!< maximum number of variables for output, diagnosis etc.
constexpr auto N_VAR_BDIAG_MAX = 10; //!< maximum number of variables for boundary diagnosis, less than N_VAR_OUTPUT_MAX because of Legion limit on return type's size

constexpr auto FILE_NAME_LEN_MAX = 128; //!< maximum length of a file name
constexpr auto VAR_NAME_LEN_MAX = 128; //!< maximum length of a variable name


using view_type_1D = Kokkos::View<rtype*>;
using host_view_type_1D = view_type_1D::HostMirror;

using view_type_2D = Kokkos::View<rtype**>;
using host_view_type_2D = view_type_2D::HostMirror;

using view_type_3D = Kokkos::View<rtype***>;
using host_view_type_3D = view_type_3D::HostMirror;

using view_type_4D = Kokkos::View<rtype****>;
using host_view_type_4D = view_type_4D::HostMirror;

using scratch_view_2D_rtype = Kokkos::View<rtype**,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using scratch_view_3D_rtype = Kokkos::View<rtype***,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using scratch_view_1D_rtype = Kokkos::View<rtype*,
        Kokkos::DefaultExecutionSpace::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

using team_policy = Kokkos::TeamPolicy<>;
using member_type = Kokkos::TeamPolicy<>::member_type;

#endif //DG_DEFINES_H
