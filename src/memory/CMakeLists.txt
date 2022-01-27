add_library(memory STATIC memory_network.cpp)
target_link_libraries(memory
    PRIVATE Kokkos::kokkoscore mpi)

set_target_properties(memory
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
