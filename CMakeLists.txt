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
    cpp/basis/nurbs.cpp
    cpp/basis/tensor.cpp
)
target_include_directories(pyck_basis PUBLIC cpp/basis)
target_link_libraries(pyck_basis PUBLIC eigen)

# Geometry library
add_library(pyck_geometry
    cpp/geometry/dof_mapper.cpp
    cpp/geometry/patch_1d.cpp
    cpp/geometry/patch_2d.cpp
    cpp/geometry/patch_2d_boundary.cpp
    cpp/geometry/factories.cpp
    cpp/geometry/evaluation.cpp
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

# Materials library (header-only)
add_library(pyck_materials INTERFACE)
target_include_directories(pyck_materials INTERFACE ${CMAKE_SOURCE_DIR}/cpp/materials)

# Assembler library
add_library(pyck_assembly
    cpp/assembly/linear_elastic_problem.cpp
)
target_include_directories(pyck_assembly PUBLIC cpp/assembly)
target_link_libraries(pyck_assembly PUBLIC eigen pyck_elements pyck_conditions pyck_constraints pyck_quadrature pyck_materials)

# Conditions library
add_library(pyck_conditions
    cpp/conditions/load_condition.cpp
    cpp/conditions/penalty_boundary_condition.cpp
    cpp/conditions/lagrange_boundary_condition.cpp
    cpp/conditions/nitsche_boundary_condition.cpp
    cpp/conditions/penalty_coupling_condition.cpp
    cpp/conditions/lagrange_coupling_condition.cpp
)
target_include_directories(pyck_conditions PUBLIC cpp/conditions)
target_link_libraries(pyck_conditions PUBLIC pyck_geometry pyck_assembly)

# Elements library
add_library(pyck_elements
    cpp/elements/beam_euler_bernoulli_1p.cpp
    cpp/elements/beam_timoshenko_1p.cpp
    cpp/elements/beam_timoshenko_2p.cpp
    cpp/elements/plate_kirchhoff_love_1p.cpp
    cpp/elements/plate_reissner_mindlin_3p.cpp
    cpp/elements/plate_reissner_mindlin_1p.cpp
)
target_include_directories(pyck_elements PUBLIC cpp/elements)
target_link_libraries(pyck_elements PUBLIC pyck_geometry pyck_assembly pyck_materials)


# pybind11
add_subdirectory(external/pybind11)

# Python module
pybind11_add_module(_pyck cpp/bindings.cpp)
target_link_libraries(_pyck PRIVATE pyck_basis pyck_geometry pyck_elements pyck_conditions pyck_constraints pyck_assembly pyck_materials)
set_target_properties(_pyck PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pyck")
install(TARGETS _pyck DESTINATION pyck)

# Test executable (unified runner)
add_executable(pyck_tests 
    tests/test_main.cpp
    tests/basis/test_knot_vector.cpp
    tests/basis/test_bspline.cpp
    tests/basis/test_nurbs.cpp
    tests/basis/test_tensor_product.cpp
    tests/geometry/test_curve.cpp
    tests/geometry/test_surface.cpp
    tests/geometry/test_dof_mapper.cpp
    tests/quadrature/test_quadrature.cpp
    tests/elements/test_euler_bernoulli.cpp
    tests/elements/test_kirchhoff_love_1p.cpp
    tests/elements/test_reissner_mindlin.cpp
    tests/elements/test_reissner_mindlin_1p.cpp
    tests/assembly/test_constraints.cpp
    tests/assembly/test_load_condition.cpp
    tests/assembly/test_materials.cpp
    tests/assembly/test_penalty_boundary_condition.cpp
    tests/assembly/test_lagrange_boundary_condition.cpp
    tests/assembly/test_multi_patch_problem.cpp
    tests/assembly/test_penalty_coupling_condition.cpp
    tests/assembly/test_lagrange_coupling_condition.cpp
)
target_link_libraries(pyck_tests PRIVATE 
    pyck_basis 
    pyck_geometry 
    pyck_elements 
    pyck_conditions 
    pyck_constraints 
    pyck_assembly 
    pyck_materials 
    pyck_quadrature 
    catch
)

enable_testing()
add_test(NAME basis_tests COMMAND pyck_tests [basis])
add_test(NAME geometry_tests COMMAND pyck_tests [geometry])
add_test(NAME quadrature_tests COMMAND pyck_tests [quadrature])
add_test(NAME assembly_tests COMMAND pyck_tests [assembly])
add_test(NAME element_tests COMMAND pyck_tests [elements])
add_test(NAME material_tests COMMAND pyck_tests [materials])
