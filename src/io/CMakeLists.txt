add_library(io STATIC io_params.cpp input_info.cpp)
target_link_libraries(io
    PUBLIC physics
    PRIVATE mesh
)
