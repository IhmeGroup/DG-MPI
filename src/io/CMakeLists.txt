# TODO: Uncomment these
add_library(io STATIC writer.cpp)#io_params.cpp input_info.cpp writer.cpp)
target_link_libraries(io
    #PUBLIC physics
    PRIVATE mesh
    PRIVATE ${HDF5_LIBRARIES}
)
target_include_directories(io
    PRIVATE ${HDF5_INCLUDE_DIRS})
