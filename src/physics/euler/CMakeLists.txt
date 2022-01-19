add_library(euler STATIC
	null.cpp
)
target_link_libraries(euler PUBLIC Kokkos::kokkoscore)
