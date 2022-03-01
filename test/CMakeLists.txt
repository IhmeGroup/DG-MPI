# -- Googletest -- #

# googletest tests
enable_testing()

# Name for the google test executable file
add_executable(
  test_gtest_all
)

# add test_<FILENAME>.cpp files that include tests for various features
# TODO: commented since they're not currently compiling on GPUs
list(APPEND GTEST_SOURCES
    main.cpp
    test_hello.cpp
    # test_physics.cpp
    math/test_linear_algebra.cpp
    numerics/basis/test_basis.cpp
    numerics/basis/test_tools.cpp
    numerics/test_nodes.cpp
    numerics/quadrature/test_quadrature.cpp
)

target_sources(test_gtest_all PRIVATE ${GTEST_SOURCES})

target_include_directories(test_gtest_all
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)

# add any directories that contain needed functions
target_link_libraries(test_gtest_all
    PRIVATE
# TODO: commented since they're not currently compiling on GPUs
        #physics
        numerics
        gtest_main
)

target_link_libraries(test_gtest_all PUBLIC Kokkos::kokkoscore)

set_target_properties(test_gtest_all
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

include(GoogleTest)
gtest_discover_tests(test_gtest_all)

# -- MPI-enabled tests -- #
add_subdirectory(mpi_enabled_tests/mesh)
add_subdirectory(mpi_enabled_tests/memory)
add_subdirectory(mpi_enabled_tests/solver)

# Name for the mpi-enabled test executable file
add_executable(test_mpi_enabled mpi_enabled_tests/main.cpp)

target_include_directories(test_mpi_enabled
    PRIVATE mesh memory
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(test_mpi_enabled
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES}
    PUBLIC Kokkos::kokkoscore mpi
    PRIVATE io
    PRIVATE test_mesh test_memory test_solver
)

set_target_properties(test_mpi_enabled
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})

# Create the directories for storing data
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/test/mpi_enabled_tests/mesh/")



# --- Sanity Checks ----- #

# name for the executable
add_executable(
    test_my_sanity
)

# add test_<FILENAME>.cpp files that include tests for various features
list(APPEND SANITY_SOURCES
    # sanity_checks/sanity_check_1.cpp
    # sanity_checks/sanity_scratchspace.cpp
    # sanity_checks/sanity_simpleparallelfor.cpp
    # sanity_checks/sanity_multiple_operators.cpp
    # sanity_checks/sanity_mult_view_by_constant.cpp
    sanity_checks/sanity_passintofunctor.cpp
)

target_sources(test_my_sanity PRIVATE ${SANITY_SOURCES})

target_link_libraries(test_my_sanity
    PUBLIC Kokkos::kokkoscore
)
