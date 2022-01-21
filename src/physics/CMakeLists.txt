add_library(physics STATIC
	euler/Date.cpp
	euler/euler.cpp
)

target_include_directories(physics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/euler")
target_include_directories(physics PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/base")

target_link_libraries(physics PUBLIC Kokkos::kokkoscore)


# add_subdirectory(base)
# add_subdirectory(euler)

# target_link_libraries(physics PUBLIC euler)