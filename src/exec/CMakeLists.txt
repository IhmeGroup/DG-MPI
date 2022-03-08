add_executable(main main.cpp)

# Note: physics is currently header only
target_link_libraries(main
    PRIVATE io solver utils numerics) # physics) 

target_include_directories(main
    PRIVATE ${METIS_INCLUDE_DIRS}
    PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(main
    PRIVATE ${METIS_LIBRARIES}
    PRIVATE ${HDF5_LIBRARIES})

# Only this target links to these libraries
if (USE_MKL)
    target_link_libraries(main PRIVATE ${MKL_LIBRARIES})
endif()

if (USE_1D)
    target_compile_definitions(main PRIVATE USE_1D)
endif()

if (USE_OLD_OUTPUT)
    target_compile_definitions(main PRIVATE USE_OLD_OUTPUT)
endif()

target_link_libraries(main PUBLIC Kokkos::kokkoscore Kokkos::kokkos)

set_target_properties(main
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
