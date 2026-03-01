cmake_minimum_required(VERSION 3.14)
project(pyck VERSION 0.1.0 LANGUAGES CXX C)

# Build Options
option(PYCK_BUILD_SINGLE_PRECISION "Build single precision (float) explicit instantiations alongside double" OFF)

if(PYCK_BUILD_SINGLE_PRECISION)
    add_compile_definitions(PYCK_BUILD_SINGLE_PRECISION)
endif()

# Set standard for C++ (C++20 required)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Eigen (header-only)
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/external/eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN_DIR})

# Catch2 (single-header v2)
add_library(catch INTERFACE)
target_include_directories(catch INTERFACE ${CMAKE_SOURCE_DIR}/external/catch)

# BSpline library
add_library(pyck_basis 
    cpp/basis/knots.cpp
    cpp/basis/bspline.cpp
    cpp/basis/tensor.cpp
)
target_include_directories(pyck_basis PUBLIC cpp/basis)
target_link_libraries(pyck_basis PUBLIC eigen)

# Geometry library
add_library(pyck_geometry
    cpp/geometry/curve.cpp
)
target_include_directories(pyck_geometry PUBLIC cpp/geometry)
target_link_libraries(pyck_geometry PUBLIC pyck_basis)

# Assembler library
add_library(pyck_assembler
    cpp/assembler/quadrature.cpp
    cpp/assembler/gauss_legendre.cpp
)
target_include_directories(pyck_assembler PUBLIC cpp/assembler
)
target_link_libraries(pyck_assembler PUBLIC eigen)


# pybind11
add_subdirectory(external/pybind11)

# Python module
pybind11_add_module(_pyck cpp/bindings.cpp)
target_link_libraries(_pyck PRIVATE pyck_geometry)
set_target_properties(_pyck PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pyck")
install(TARGETS _pyck DESTINATION pyck)

# Test executables
add_executable(test_knot_vector tests/test_knot_vector.cpp)
target_link_libraries(test_knot_vector PRIVATE pyck_basis catch)

add_executable(test_bspline tests/test_bspline.cpp)
target_link_libraries(test_bspline PRIVATE pyck_basis catch)

add_executable(test_tensor_product tests/test_tensor_product.cpp)
target_link_libraries(test_tensor_product PRIVATE pyck_basis catch)

add_executable(test_curve tests/test_curve.cpp)
target_link_libraries(test_curve PRIVATE pyck_geometry catch)

add_executable(test_quadrature tests/test_quadrature.cpp)
target_link_libraries(test_quadrature PRIVATE pyck_assembler catch)

enable_testing()
add_test(NAME knot_vector_tests COMMAND test_knot_vector)
add_test(NAME bspline_tests COMMAND test_bspline)
add_test(NAME tensor_product_tests COMMAND test_tensor_product)
add_test(NAME curve_tests COMMAND test_curve)
add_test(NAME quadrature_tests COMMAND test_quadrature)
