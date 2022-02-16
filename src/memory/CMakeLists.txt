add_library(memory STATIC memory_network.h)
target_link_libraries(memory
    PUBLIC Kokkos::kokkoscore
    PRIVATE mpi mesh)

set_target_properties(memory
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
