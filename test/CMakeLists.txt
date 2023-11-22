include_directories(../)
file(
    GLOB GTEST_CPP_FILES
    "test_params.cpp"
    "test_index.cpp"
    "test_index_recall.cpp"
)

add_executable(runtest)

target_sources(runtest PUBLIC ${GTEST_CPP_FILES})

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

target_link_libraries(runtest PUBLIC
    puck
    gtest_main
)
target_link_libraries(runtest PUBLIC
        ${MKL_LIBRARIES}/libmkl_intel_lp64.a
        ${MKL_LIBRARIES}/libmkl_sequential.a
        ${MKL_LIBRARIES}/libmkl_core.a
)

gtest_discover_tests(runtest)


include_directories(../)
add_executable(my_benchmark my_benchmark.cpp)

target_link_libraries(my_benchmark PUBLIC
    puck
)

set(GTEST_TOOLS_PREFIX ${CMAKE_INSTALL_PREFIX}/test)
install(TARGETS runtest my_benchmark DESTINATION ${GTEST_TOOLS_PREFIX})
