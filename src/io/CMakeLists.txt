# TODO: Uncomment these
add_library(io STATIC writer.cpp)#io_params.cpp input_info.cpp writer.cpp)

target_include_directories(io
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(io
    PUBLIC solver
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES}
    PUBLIC Kokkos::kokkoscore
)

set_target_properties(io
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
