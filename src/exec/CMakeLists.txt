add_executable(main main.cpp)

# TODO: Uncomment
target_link_libraries(main
    PRIVATE io memory mesh solver utils)# numerics

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

# Notes:
# it is important (I think this is the minimal requirement) that all Kokkos libraries are built as static
# otherwise they will bring with them at link time the system omp library (see note about omp flags)
target_link_libraries(main PRIVATE Kokkos::kokkoscore)
# prevent from linking to system's OpenMP implementation
# target_link_options(main PRIVATE "-Wl,--as-needed")


# Notes:
# 1) if omp symbols are missing, ask Sean to add them to realm's omp implementation
#    in particular, don't try to link with any omp library such as libgomp or libiommp
#    this will conflicts with realm's omp implementation
#    we still need the appropriate omp flags but ONLY at compile-time
# 2) I don't remember the role of ENABLE_EXPORTS ON
#    I think this has to do with debugging and getting line numbers with stack traces
set_target_properties(main
    PROPERTIES
        ENABLE_EXPORTS ON
        COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
