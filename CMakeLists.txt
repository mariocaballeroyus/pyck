cmake_minimum_required(VERSION 3.14)
project(pyck_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen (header-only)
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/external/eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN_DIR})

# Catch2 (single-header v2)
add_library(catch INTERFACE)
target_include_directories(catch INTERFACE ${CMAKE_SOURCE_DIR}/external/catch)

# BSpline library
add_library(pyck_basis
    cpp/basis/bspline.cpp
)
target_include_directories(pyck_basis PUBLIC
    cpp/basis
)
target_link_libraries(pyck_basis PUBLIC eigen)

# Test executable
add_executable(test_bspline
    tests/test_bspline.cpp
)
target_link_libraries(test_bspline PRIVATE pyck_basis catch)

enable_testing()
add_test(NAME bspline_tests COMMAND test_bspline)
