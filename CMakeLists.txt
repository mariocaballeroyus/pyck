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
    cpp/geometry/dof_mapper.cpp
    cpp/geometry/curve.cpp
    cpp/geometry/surface.cpp
    cpp/geometry/boundary_patch.cpp
)
target_include_directories(pyck_geometry PUBLIC cpp/geometry)
target_link_libraries(pyck_geometry PUBLIC pyck_basis pyck_quadrature)

# Constraints library
add_library(pyck_constraints
    cpp/constraints/direct_constraint.cpp
    cpp/constraints/linear_constraint.cpp
)
target_include_directories(pyck_constraints PUBLIC cpp/constraints)
target_link_libraries(pyck_constraints PUBLIC eigen)

# Quadrature library
add_library(pyck_quadrature
    cpp/quadrature/quadrature.cpp
    cpp/quadrature/gauss_legendre.cpp
)
target_include_directories(pyck_quadrature PUBLIC cpp/quadrature)
target_link_libraries(pyck_quadrature PUBLIC eigen)

# Assembler library
add_library(pyck_assembly
    cpp/assembly/linear_elastic_problem.cpp
)
target_include_directories(pyck_assembly PUBLIC cpp/assembly)
target_link_libraries(pyck_assembly PUBLIC eigen pyck_elements pyck_conditions pyck_constraints pyck_quadrature)

# Conditions library
add_library(pyck_conditions
    cpp/conditions/load_condition.cpp
)
target_include_directories(pyck_conditions PUBLIC cpp/conditions)
target_link_libraries(pyck_conditions PUBLIC pyck_geometry pyck_assembly)

# Elements library
add_library(pyck_elements
    cpp/elements/euler_bernoulli_beam_1p.cpp
    cpp/elements/timoshenko_beam_1p.cpp
    cpp/elements/timoshenko_beam_2p.cpp
)
target_include_directories(pyck_elements PUBLIC cpp/elements)
target_link_libraries(pyck_elements PUBLIC pyck_geometry pyck_assembly)


# pybind11
add_subdirectory(external/pybind11)

# Python module
pybind11_add_module(_pyck cpp/bindings.cpp)
target_link_libraries(_pyck PRIVATE pyck_basis pyck_geometry pyck_elements pyck_conditions pyck_constraints pyck_assembly)
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

add_executable(test_dof_mapper tests/test_dof_mapper.cpp)
target_link_libraries(test_dof_mapper PRIVATE pyck_geometry catch)

add_executable(test_load_condition tests/test_load_condition.cpp)
target_link_libraries(test_load_condition PRIVATE pyck_conditions pyck_assembly pyck_basis catch)

add_executable(test_quadrature tests/test_quadrature.cpp)
target_link_libraries(test_quadrature PRIVATE pyck_quadrature catch)

add_executable(test_euler_bernoulli tests/test_euler_bernoulli.cpp)
target_link_libraries(test_euler_bernoulli PRIVATE pyck_elements pyck_conditions pyck_constraints pyck_assembly pyck_geometry pyck_basis catch)

add_executable(test_constraints tests/test_constraints.cpp)
target_link_libraries(test_constraints PRIVATE pyck_constraints pyck_assembly catch)

add_executable(test_surface tests/test_surface.cpp)
target_link_libraries(test_surface PRIVATE pyck_geometry catch)

enable_testing()
add_test(NAME knot_vector_tests COMMAND test_knot_vector)
add_test(NAME bspline_tests COMMAND test_bspline)
add_test(NAME tensor_product_tests COMMAND test_tensor_product)
add_test(NAME curve_tests COMMAND test_curve)
add_test(NAME quadrature_tests COMMAND test_quadrature)
add_test(NAME dof_mapper_tests COMMAND test_dof_mapper)
add_test(NAME load_condition_tests COMMAND test_load_condition)
add_test(NAME euler_bernoulli_tests COMMAND test_euler_bernoulli)
add_test(NAME constraints_tests COMMAND test_constraints)
add_test(NAME surface_tests COMMAND test_surface)
