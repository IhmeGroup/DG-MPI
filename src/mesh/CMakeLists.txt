add_library(mesh STATIC mesh.cpp)
target_include_directories(mesh
    PRIVATE "${METIS_ROOT}/include"
    PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(mesh
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES})
