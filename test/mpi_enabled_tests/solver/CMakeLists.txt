add_library(test_solver STATIC test_helpers.cpp)
target_include_directories(test_solver
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(test_solver
    PUBLIC Kokkos::kokkoscore
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES}
    PRIVATE mpi
    PRIVATE numerics
    PRIVATE solver
)

#set_target_properties(mesh
#    PROPERTIES
#        ENABLE_EXPORTS ON
#        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
