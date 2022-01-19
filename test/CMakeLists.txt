# googletest tests
enable_testing()

# name for the google test executable file
add_executable(
  test_gtest_all
)

# add test_<FILENAME>.cpp files that include tests for various features
list(APPEND GTEST_SOURCES
    test_hello.cpp
    test_physics.cpp
)

target_sources(test_gtest_all PRIVATE ${GTEST_SOURCES})

# add any directories that contain needed functions
target_link_libraries(test_gtest_all
    PRIVATE
        # mesh physics numerics equations
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_gtest_all)