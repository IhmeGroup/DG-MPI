# -- Googletest -- #

# googletest tests
enable_testing()

# Name for the google test executable file
add_executable(
  test_gtest_all
)

# add test_<FILENAME>.cpp files that include tests for various features
list(APPEND GTEST_SOURCES
    main.cpp
    test_hello.cpp
    test_physics.cpp
    numerics/basis/test_basis.cpp
    numerics/basis/test_tools.cpp
    numerics/test_nodes.cpp
    numerics/quadrature/test_quadrature.cpp
)

target_sources(test_gtest_all PRIVATE ${GTEST_SOURCES})

# add any directories that contain needed functions
target_link_libraries(test_gtest_all
    PRIVATE
        physics
        numerics
        gtest_main
)

target_link_libraries(test_gtest_all PUBLIC Kokkos::kokkoscore)

include(GoogleTest)
gtest_discover_tests(test_gtest_all)

# -- MPI-enabled tests -- #

# Name for the mpi-enabled test executable file
add_executable(
  test_mpi_enabled
)

# add test_<FILENAME>.cpp files that include tests for various features
list(APPEND MPITEST_SOURCES
    mpi_enabled_tests/main.cpp
    mpi_enabled_tests/mesh/test_mesh.cpp
)

target_sources(test_mpi_enabled PRIVATE ${MPITEST_SOURCES})

# add any directories that contain needed functions
target_link_libraries(test_mpi_enabled
    PRIVATE
        mesh
)

target_link_libraries(test_mpi_enabled
    PUBLIC Kokkos::kokkoscore mpi)

# Add source directory for use in finding test input files
add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")
