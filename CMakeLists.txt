# TODO: There are a few things in here that are either no longer needed or need
# to be further understood. Some obvious things have been removed so far
# (anything related to Legion and Eigen).

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW) # use <PackageName>_ROOT over <PackageName>_DIR
# Set the default build type to be debug for now
set(CMAKE_BUILD_TYPE Release)

# Set dynamic library name depending on operating system
if(APPLE)
    set(so "dylib")
else()
    set(so "so")
endif()

# profile build time
# set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")

# run path for installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# some variables to help find external libraries.
if(GPU_BUILD)
    set(METIS_ROOT "${CMAKE_SOURCE_DIR}/build_gpu/build_externals/metis/install" CACHE PATH "METIS root")
else()
    set(METIS_ROOT "${CMAKE_SOURCE_DIR}/build_cpu/build_externals/metis/install" CACHE PATH "METIS root")
endif()

#if(DEFINED ENV{HDF5_DIR})
#    message(STATUS "HDF5 obtained via environment")
#    set(HDF5_ROOT "$ENV{HDF5_DIR}")
#else()
message(STATUS "HDF5 obtained via dg-mpi/externals folder")
set(HDF5_ROOT "${CMAKE_SOURCE_DIR}/externals/hdf5-download/install/" CACHE PATH "HDF5 root")
message(STATUS "${CMAKE_SOURCE_DIR}/externals/hdf5_download/install")
#endif()

if(DEFINED ENV{MPI_DIR})
    set(MPI_ROOT "$ENV{MPI_DIR}")
else()
    message(STATUS "Using local installs for mpich")
endif()

if(GPU_BUILD)
    set(Kokkos_ROOT         "${CMAKE_SOURCE_DIR}/build_gpu/build_externals/kokkos/build/install" CACHE PATH "Kokkos root")
    set(KokkosKernels_ROOT  "${CMAKE_SOURCE_DIR}/build_gpu/build_externals/kokkos-kernels/build/install" CACHE PATH "Kokkos-kernels root")
else()
    set(Kokkos_ROOT         "${CMAKE_SOURCE_DIR}/build_cpu/build_externals/kokkos/build/install" CACHE PATH "Kokkos root")
    set(KokkosKernels_ROOT  "${CMAKE_SOURCE_DIR}/build_cpu/build_externals/kokkos-kernels/build/install" CACHE PATH "Kokkos-kernels root")
endif()

# Use the MPI compilers
if(NOT APPLE)
    # set(CMAKE_C_COMPILER "${Kokkos_ROOT}/bin/kokkos_launch_compiler")
    # set(CMAKE_CXX_COMPILER "${Kokkos_ROOT}/bin/kokkos_launch_compiler")
    if(DEFINED ENV{MPI_DIR})
       set(CMAKE_C_COMPILER "${MPI_ROOT}/bin/mpicc")
       set(CMAKE_CXX_COMPILER "${MPI_ROOT}/bin/mpicxx")
    else()
       set(CMAKE_C_COMPILER "mpicc")
       set(CMAKE_CXX_COMPILER "mpicxx")
    endif()
endif()

# Trying to make MPI CUDA-aware
# TODO: This doesn't really do anything because MPI needs to be compiled with
# CUDA awareness
set(MPICH_CXX "${CMAKE_SOURCE_DIR}/externals/kokkos/build/install/bin/nvcc_wrapper")


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

# Sets the number of species if using more than one. 
# Defaults to 1 (or air)
# If user increases above 1, user is responsible for either
# cleaning the cache or specifically setting the NUM_SPECIES var 
# back to 1 if they wish to go back to the default build. 
if(DEFINED NUM_SPECIES)
    set(CMAKE_NUM_SPECIES ${NUM_SPECIES})
else()
    set(CMAKE_NUM_SPECIES 1)
endif()

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

set(HDF5_LIBRARIES "${HDF5_ROOT}/lib/libhdf5.a")
if(NOT HDF5_LIBRARIES)
    message(FATAL_ERROR "The HDF5 library could not be found. Please provide -DHDF5_ROOT.")
endif()

# MPI
include_directories("${MPI_ROOT}/include")
link_directories("${MPI_ROOT}/lib/libmpich.${so}")

# Kokkos
include_directories("${Kokkos_ROOT}/include")
include_directories("${KokkosKernels_ROOT}/include")

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

# Specify compiler information. This has to be done after finding Kokkos, since
# Kokkos modifies these.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
if(GPU_BUILD)
    set(warning_flags "-Wall -Wextra -Wno-deprecated-gpu-targets")
else()
    set(warning_flags "-Wall -Wextra")
endif()
set(CMAKE_C_FLAGS_DEBUG      "-g  -DDEBUG -O0 ${warning_flags}")
set(CMAKE_CXX_FLAGS_DEBUG    "-g  -DDEBUG -O0 ${warning_flags}")
set(CMAKE_CUDA_FLAGS_DEBUG   "-g  -DDEBUG -O0 ${warning_flags}")
set(CMAKE_C_FLAGS_RELEASE    "-O3 ${warning_flags}")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 ${warning_flags}")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 ${warning_flags}")
set(CMAKE_EXE_LINKER_FLAGS "-lz")

# set preprocessor definitions
if(USE_DOUBLES)
    add_compile_definitions(USE_DOUBLES)
endif()
add_compile_definitions(CMAKE_NUM_SPECIES)

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
add_subdirectory(src/numerics)
add_subdirectory(src/physics)
add_subdirectory(src/mesh)
add_subdirectory(src/solver)
add_subdirectory(src/utils)
add_subdirectory(test)
add_subdirectory(src/post)
# executables
add_subdirectory(src/exec)

###########
# summary #
###########

message(STATUS "-----------------------------------------------------------")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "METIS root:          ${METIS_ROOT}")
message(STATUS "METIS include:       ${METIS_INCLUDE_DIRS}")
message(STATUS "METIS lib:           ${METIS_LIBRARIES}")
message(STATUS "HDF5 root:           ${HDF5_ROOT}")
message(STATUS "Kokkos root:         ${Kokkos_ROOT}")
message(STATUS "MPI root:            ${MPI_ROOT}")
message(STATUS "NUMBER OF SPECIES:   ${CMAKE_NUM_SPECIES}")
message(STATUS "")
    # message(STATUS "Kokkos kernels root: ${KokkosKernels_ROOT}")

# if(BUILD_BENCHMARKS)
    # message(STATUS "sltbench root:       ${SLTBENCH_ROOT}")
# endif()

message(STATUS "")
message(STATUS "CMAKE_CXX_COMPILER:      ${CMAKE_CXX_COMPILER}")
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
