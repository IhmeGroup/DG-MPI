add_library(solver STATIC 
    base.cpp )

target_link_libraries(solver
    PRIVATE Kokkos::kokkoscore
    PRIVATE memory
    PRIVATE numerics)

set_target_properties(solver
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
