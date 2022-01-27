add_library(numerics STATIC
    numerics_data.cpp
    basis/basis.cpp
    basis/tools.cpp
    )

target_include_directories(numerics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/basis")

target_link_libraries(numerics PUBLIC Kokkos::kokkoscore
    PUBLIC Kokkos::kokkoskernels)
