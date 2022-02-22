add_library(test_memory STATIC test_memory_network.cpp)
target_include_directories(test_memory
    PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(test_memory
    PUBLIC Kokkos::kokkoscore
    PRIVATE ${HDF5_LIBRARIES}
    PRIVATE mpi mesh)

set_target_properties(test_memory
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
