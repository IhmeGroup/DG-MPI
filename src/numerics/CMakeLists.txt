add_library(numerics STATIC
    numerics_data.cpp
    nodes.cpp
    basis/basis.cpp
    basis/shape.cpp
    basis/tools.cpp
    quadrature/segments.cpp
    quadrature/quadrilateral.cpp
    quadrature/hexahedron.cpp
    )

target_include_directories(numerics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/basis")

target_link_libraries(numerics PUBLIC Kokkos::kokkoscore
    PUBLIC Kokkos::kokkoskernels)
