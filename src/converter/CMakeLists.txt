add_executable(converter main.cpp)

target_link_libraries(converter
    PRIVATE io solver utils numerics)

target_include_directories(converter
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(converter
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES})

target_link_libraries(converter PUBLIC Kokkos::kokkoscore Kokkos::kokkos)
