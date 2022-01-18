# TODO: There are a few things in here that are either no longer needed or need
# to be further understood. Some obvious things have been removed so far
# (anything related to Legion and Eigen).

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW) # use <PackageName>_ROOT over <PackageName>_DIR
project(DG VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
set(HDF5_ROOT           "${CMAKE_SOURCE_DIR}/externals/hdf5/build/build" CACHE PATH "HDF5 root")
set(Kokkos_ROOT         CACHE PATH "Kokkos root")
set(KokkosKernels_ROOT  CACHE PATH "KokkosKernels root")
set(SLTBENCH_ROOT       CACHE PATH "sltbench root")
set(CANTERA_ROOT        CACHE PATH "Cantera root")

# specify default cmake options
option(USE_DOUBLES      "Use double precision" ON)
option(USE_OMP          "Use OpenMP" ON)
option(USE_DCR          "Use dynamic control replication" ON)
option(USE_MS_CLIPPING  "Clip negative mass fractions in the physics object" ON)
option(USE_OLD_OUTPUT   "Use old output info at every time-step" ON)
option(USE_1D           "Trick to fake 1D calculation" OFF)
option(USE_MKL          "Use MKL libraries" OFF)
option(USE_KOKKOS       "Use Kokkos" OFF)
option(BUILD_TESTS      "Build the tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_EXAMPLES   "Build examples" OFF)

# MKL libraries to link against
set(MKL_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core)

# Hard external requirements
# METIS
find_library(METIS_LIBRARIES metis HINTS "${METIS_ROOT}/lib")
if(NOT METIS_LIBRARIES)
    message(FATAL_ERROR "The METIS library could not be found. Please provide -DMETIS_ROOT.")
endif()
# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX PATHS "${HDF5_ROOT}/share/cmake")


if (USE_KOKKOS)
    find_package(Kokkos REQUIRED)
    find_package(KokkosKernels REQUIRED)
    add_compile_definitions(DG_USE_KOKKOS)
endif()
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
if (USE_MKL)
    add_compile_definitions(ENABLE_EIGEN_MKL)
endif()
if(USE_DCR)
    add_compile_definitions(USE_DCR)
endif()

# global include
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/externals")
include_directories(${EIGEN_ROOT})

# temporary fix for compiling on MacOS
# the first step is to install llvm and libomp via homebrew
if(APPLE)
    include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
    link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
endif()

##############
# googletest #
##############

if(BUILD_TESTS)
    # Download and unpack googletest at configure time
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                     EXCLUDE_FROM_ALL)

    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
      include_directories("${gtest_SOURCE_DIR}/include")
    endif()
endif()

###########################
# project sub-directories #
###########################

# building blocks
add_subdirectory(src/mesh)
add_subdirectory(src/utils)

# executables
add_subdirectory(src/exec)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

###########
# summary #
###########

message(STATUS "-----------------------------------------------------------")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "METIS root:          ${METIS_ROOT}")
message(STATUS "HDF5 root:           ${HDF5_ROOT}")
if (USE_KOKKOS)
    message(STATUS "Kokkos root:         ${Kokkos_ROOT}")
    message(STATUS "Kokkos kernels root: ${KokkosKernels_ROOT}")
endif()
if(BUILD_BENCHMARKS)
    message(STATUS "sltbench root:       ${SLTBENCH_ROOT}")
endif()

message(STATUS "")
message(STATUS "CMAKE_CXX_FLAGS:         ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:   ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

message(STATUS "")
message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
if(USE_MKL)
    message(STATUS "")
    message(STATUS "MKL enabled")
endif()
message(STATUS "-----------------------------------------------------------")
