add_executable(post main.cpp)

target_link_libraries(post
    PRIVATE io solver utils numerics)

target_include_directories(post
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(post
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES})

target_link_libraries(post PUBLIC Kokkos::kokkoscore Kokkos::kokkos)