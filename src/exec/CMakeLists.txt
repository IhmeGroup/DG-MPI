# main solver

add_executable(solver main.cpp)

target_link_libraries(solver
    PRIVATE mesh)

# Only this target links to these libraries
if (USE_MKL)
    target_link_libraries(solver PRIVATE ${MKL_LIBRARIES})
endif()

if (USE_1D)
    target_compile_definitions(solver PRIVATE USE_1D)
endif()

if (USE_OLD_OUTPUT)
    target_compile_definitions(solver PRIVATE USE_OLD_OUTPUT)
endif()

# Notes:
# it is important (I think this is the minimal requirement) that all Kokkos libraries are built as static
# otherwise they will bring with them at link time the system omp library (see note about omp flags)
if(USE_KOKKOS)
    target_link_libraries(solver PRIVATE Kokkos::kokkoscore)
    # prevent from linking to system's OpenMP implementation
    target_link_options(solver PRIVATE "-Wl,--as-needed")
endif()

# Notes:
# 1) if omp symbols are missing, ask Sean to add them to realm's omp implementation
#    in particular, don't try to link with any omp library such as libgomp or libiommp
#    this will conflicts with realm's omp implementation
#    we still need the appropriate omp flags but ONLY at compile-time
# 2) I don't remember the role of ENABLE_EXPORTS ON
#    I think this has to do with debugging and getting line numbers with stack traces
set_target_properties(solver
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
