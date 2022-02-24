add_library(test_mesh STATIC test_mesh.cpp)
target_include_directories(test_mesh
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(test_mesh
    PRIVATE io
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES}
)

#set_target_properties(mesh
#    PROPERTIES
#        ENABLE_EXPORTS ON
#        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
