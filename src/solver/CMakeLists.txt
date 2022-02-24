add_library(solver STATIC
    base.cpp )

target_include_directories(solver
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(solver
    PUBLIC Kokkos::kokkoscore
    PRIVATE numerics)

set_target_properties(solver
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
