#add_library(mesh STATIC mesh.cpp)
#target_include_directories(mesh
#    PRIVATE ${METIS_INCLUDE_DIRS}
#    PRIVATE ${HDF5_INCLUDE_DIRS})
#target_link_libraries(mesh
#    PRIVATE ${METIS_LIBRARIES}
#    PRIVATE ${HDF5_LIBRARIES}
#    PUBLIC Kokkos::kokkoscore
#    PUBLIC Kokkos::kokkos
#    PRIVATE numerics)
#
#set_target_properties(mesh
#    PROPERTIES
#        ENABLE_EXPORTS ON
#        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
