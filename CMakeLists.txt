# TODO: There are a few things in here that are either no longer needed or need
# to be further understood. Some obvious things have been removed so far
# (anything related to Legion and Eigen).

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW) # use <PackageName>_ROOT over <PackageName>_DIR
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set dynamic library name depending on operating system
if(APPLE)
    set(so "dylib")
else()
    set(so "so")
endif()

# profile build time
# set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

# specify build type specific flags
set(warning_flags "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG   "-g  -O0 ${warning_flags}")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG ${warning_flags}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${warning_flags}")

# run path for installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# some variables to help find external libraries
set(METIS_ROOT          "${CMAKE_SOURCE_DIR}/externals/metis/build/install" CACHE PATH "METIS root")

if(DEFINED ENV{HDF5_DIR})
    message(STATUS "HDF5 obtained via environment")
    set(HDF5_ROOT "$ENV{HDF5_DIR}")
else()
    message(STATUS "HDF5 obtained via dg-mpi/externals folder")
    set(HDF5_ROOT "${CMAKE_SOURCE_DIR}/externals/hdf5/build/install")
endif()

if(DEFINED ENV{MPI_DIR})
    set(MPI_ROOT "$ENV{MPI_DIR}")
else()
    message(STATUS "Using local installs for mpich")
endif()

set(Kokkos_ROOT         "${CMAKE_SOURCE_DIR}/externals/kokkos/build/install" CACHE PATH "Kokkos root")
set(KokkosKernels_ROOT  "${CMAKE_SOURCE_DIR}/externals/kokkos-kernels/build/install" CACHE PATH "Kokkos-kernels root")

# Use the MPI compilers
if(NOT APPLE)
    if(DEFINED ENV{MPI_DIR})
        set(CMAKE_C_COMPILER "${MPI_ROOT}/bin/mpicc")
        set(CMAKE_CXX_COMPILER "${MPI_ROOT}/bin/mpicxx")
    else()
        set(CMAKE_C_COMPILER "mpicc")
        set(CMAKE_CXX_COMPILER "mpicxx")
    endif()
endif()


# Set the project info (done after setting compilers to avoid infinite loop)
project(DG VERSION 1.0)

# specify default cmake options
option(USE_DOUBLES      "Use double precision" ON)
option(USE_OMP          "Use OpenMP" ON)
# option(USE_MS_CLIPPING  "Clip negative mass fractions in the physics object" ON)
# option(USE_OLD_OUTPUT   "Use old output info at every time-step" ON)
# option(USE_1D           "Trick to fake 1D calculation" OFF)
# option(USE_MKL          "Use MKL libraries" OFF)
# option(USE_KOKKOS       "Use Kokkos" OFF)
# option(BUILD_TESTS      "Build the tests" OFF)
# option(BUILD_BENCHMARKS "Build benchmarks" OFF)
# option(BUILD_EXAMPLES   "Build examples" OFF)

# MKL libraries to link against
# set(MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)


# -- Hard external requirements -- #
# METIS
set(METIS_INCLUDE_DIRS "${METIS_ROOT}/include")
set(METIS_LIBRARIES "${METIS_ROOT}/lib/libmetis.a")
if(NOT METIS_LIBRARIES)
    message(FATAL_ERROR "The METIS library could not be found. Please provide -DMETIS_ROOT.")
endif()

# HDF5
set(HDF5_INCLUDE_DIRS "${HDF5_ROOT}/include")
set(HDF5_LIBRARIES "${HDF5_ROOT}/lib/libhdf5_cpp.${so};${HDF5_ROOT}/lib/libhdf5.${so}")
if(NOT HDF5_LIBRARIES)
    message(FATAL_ERROR "The HDF5 library could not be found. Please provide -DHDF5_ROOT.")
endif()

# MPI
include_directories("${MPI_ROOT}/include")
link_directories("${MPI_ROOT}/lib/libmpich.${so}")

##############
# googletest #
##############
# ----------------------------------------------------------------------------------------------- #
# For the googletest dependency we use CMakes googletest module.
# For more information see the following links:
#
#   Googletest github: https://github.com/google/googletest
#   Googletest + cmake tutorial: https://google.github.io/googletest/quickstart-cmake.html
#   CMake Googletest module: https://cmake.org/cmake/help/git-stage/module/GoogleTest.html
#
# ----------------------------------------------------------------------------------------------- #

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929 # release-1.11.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# ----------------------------------------------------------------------------------------------- #



find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(OpenMP REQUIRED)



# set preprocessor definitions
if(USE_DOUBLES)
    add_compile_definitions(USE_DOUBLES)
endif()

# global include
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/externals")

# temporary fix for compiling on MacOS
# the first step is to install llvm and libomp via homebrew
if(APPLE)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

# Add source directory for use in the code (for example, for finding input files
# for tests)
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

###########################
# project sub-directories #
###########################

# building blocks
add_subdirectory(src/io)
add_subdirectory(src/memory)
add_subdirectory(src/mesh)
add_subdirectory(src/numerics)
add_subdirectory(src/physics)
add_subdirectory(src/solver)
add_subdirectory(src/utils)
add_subdirectory(test)

# executables
add_subdirectory(src/exec)

###########
# summary #
###########

message(STATUS "-----------------------------------------------------------")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "METIS root:          ${METIS_ROOT}")
message(STATUS "HDF5 root:           ${HDF5_ROOT}")
message(STATUS "Kokkos root:         ${Kokkos_ROOT}")
message(STATUS "MPI root:            ${MPI_ROOT}")
    # message(STATUS "Kokkos kernels root: ${KokkosKernels_ROOT}")

# if(BUILD_BENCHMARKS)
    # message(STATUS "sltbench root:       ${SLTBENCH_ROOT}")
# endif()

message(STATUS "")
message(STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

message(STATUS "")
message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
message(STATUS "")
# if(USE_MKL)
    # message(STATUS "")
    # message(STATUS "MKL enabled")
# endif()
message(STATUS "-----------------------------------------------------------")
