add_library(base STATIC
	null.cpp
)
target_link_libraries(base PUBLIC Kokkos::kokkoscore)
