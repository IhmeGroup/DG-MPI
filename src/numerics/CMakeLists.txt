add_library(numerics STATIC
    numerics_data.cpp
    nodes.cpp
    basis/basis.cpp
    basis/shape.cpp
    quadrature/tools.cpp
    quadrature/segment.cpp
    quadrature/quadrilateral.cpp
    quadrature/hexahedron.cpp
    )

target_include_directories(numerics
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(numerics PUBLIC Kokkos::kokkoscore
    PUBLIC Kokkos::kokkoskernels)
