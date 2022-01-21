# TODO: There are a few things in here that are either no longer needed or need
# to be further understood. Some obvious things have been removed so far
# (anything related to Legion and Eigen).

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW) # use <PackageName>_ROOT over <PackageName>_DIR
project(DG VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
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
set(CMAKE_CXX_FLAGS_DEBUG   "-g  -O0 -Wall")
set(CMAKE_C_FLAGS_RELEASE   "-O3 -DNDEBUG -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

# run path for installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# some variables to help find external libraries
set(METIS_ROOT          "${CMAKE_SOURCE_DIR}/externals/metis/build/install" CACHE PATH "METIS root")
set(HDF5_ROOT           "${CMAKE_SOURCE_DIR}/externals/hdf5/build/install" CACHE PATH "HDF5 root")
set(Kokkos_ROOT         "${CMAKE_SOURCE_DIR}/externals/kokkos/build/install" CACHE PATH "Kokkos root")
# set(KokkosKernels_ROOT  CACHE PATH "KokkosKernels root")
# set(SLTBENCH_ROOT       CACHE PATH "sltbench root")
# set(CANTERA_ROOT        CACHE PATH "Cantera root")

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
#   Using the version from the following link (using the *.zip file at the bottom of page):
#   https://github.com/google/googletest/releases/tag/release-1.11.0
# ----------------------------------------------------------------------------------------------- #

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
# ----------------------------------------------------------------------------------------------- #



# if (USE_KOKKOS)
find_package(Kokkos REQUIRED)
# find_package(KokkosKernels REQUIRED)
    # add_compile_definitions(DG_USE_KOKKOS)
# endif()
# Note: do not use target_link_libraries(target PUBLIC OpenMP::OpenMP_CXX)
# This will link against libgomp and pthread with GNU.
# You don't want that: Realm is providing the OpenMP runtime.
# The -fopenmp flag is still required at compile time.
# If symbols are missing, report it to Sean Treichler.
find_package(OpenMP REQUIRED)

# set preprocessor definitions
if(USE_DOUBLES)
    add_compile_definitions(USE_DOUBLES)
endif()
# if (USE_MKL)
    # add_compile_definitions(ENABLE_EIGEN_MKL)
# endif()
# if(USE_DCR)
    # add_compile_definitions(USE_DCR)
# endif()

# global include
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/externals")
# include_directories(${EIGEN_ROOT})

# temporary fix for compiling on MacOS
# the first step is to install llvm and libomp via homebrew
if(APPLE)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()


###########################
# project sub-directories #
###########################

# building blocks
add_subdirectory(src/io)
add_subdirectory(src/physics)
add_subdirectory(src/mesh)
add_subdirectory(src/numerics)
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
