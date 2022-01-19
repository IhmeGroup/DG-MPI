add_library(physics STATIC
	null.cpp
)
target_link_libraries(physics PUBLIC Kokkos::kokkoscore)


add_subdirectory(base)
add_subdirectory(euler)
