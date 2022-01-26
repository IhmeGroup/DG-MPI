add_library(memory STATIC memory_network.cpp)
target_link_libraries(memory
    PRIVATE mpi)
