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
file(GLOB PYCK_BASIS_SOURCES cpp/basis/*.cpp)
add_library(pyck_basis ${PYCK_BASIS_SOURCES})
target_include_directories(pyck_basis PUBLIC cpp/basis)
target_link_libraries(pyck_basis PUBLIC eigen)

# Geometry library
file(GLOB PYCK_GEOMETRY_SOURCES cpp/geometry/*.cpp)
add_library(pyck_geometry ${PYCK_GEOMETRY_SOURCES})
target_include_directories(pyck_geometry PUBLIC cpp/geometry cpp/elements)
target_link_libraries(pyck_geometry PUBLIC pyck_basis pyck_quadrature)

# Operators library (differential-geometry operators on IGA fields)
file(GLOB PYCK_OPERATORS_SOURCES cpp/operators/*.cpp)
add_library(pyck_operators ${PYCK_OPERATORS_SOURCES})
target_include_directories(pyck_operators PUBLIC cpp/operators)
target_link_libraries(pyck_operators PUBLIC eigen)

# Constraints library
file(GLOB PYCK_CONSTRAINTS_SOURCES cpp/constraints/*.cpp)
add_library(pyck_constraints ${PYCK_CONSTRAINTS_SOURCES})
target_include_directories(pyck_constraints PUBLIC cpp/constraints)
target_link_libraries(pyck_constraints PUBLIC eigen)

# Quadrature library
file(GLOB PYCK_QUADRATURE_SOURCES cpp/quadrature/*.cpp)
add_library(pyck_quadrature ${PYCK_QUADRATURE_SOURCES})
target_include_directories(pyck_quadrature PUBLIC cpp/quadrature)
target_link_libraries(pyck_quadrature PUBLIC eigen)

# Materials library (header-only)
add_library(pyck_materials INTERFACE)
target_include_directories(pyck_materials INTERFACE ${CMAKE_SOURCE_DIR}/cpp/materials)

# Assembler library
file(GLOB PYCK_ASSEMBLY_SOURCES cpp/assembly/*.cpp)
add_library(pyck_assembly ${PYCK_ASSEMBLY_SOURCES})
target_include_directories(pyck_assembly PUBLIC cpp/assembly)
target_link_libraries(pyck_assembly PUBLIC eigen pyck_elements pyck_conditions pyck_constraints pyck_quadrature pyck_materials)

# Conditions library
file(GLOB PYCK_CONDITIONS_SOURCES cpp/conditions/*.cpp)
add_library(pyck_conditions ${PYCK_CONDITIONS_SOURCES})
target_include_directories(pyck_conditions PUBLIC cpp/conditions)
target_link_libraries(pyck_conditions PUBLIC pyck_geometry pyck_assembly)

# Postprocessing library (header-only)
add_library(pyck_postprocessing INTERFACE)
target_include_directories(pyck_postprocessing INTERFACE cpp/postprocessing)
target_link_libraries(pyck_postprocessing INTERFACE pyck_geometry pyck_quadrature)

# Elements library
file(GLOB PYCK_ELEMENTS_SOURCES cpp/elements/*.cpp)
add_library(pyck_elements ${PYCK_ELEMENTS_SOURCES})
target_include_directories(pyck_elements PUBLIC cpp/elements)
target_link_libraries(pyck_elements PUBLIC pyck_geometry pyck_operators pyck_assembly pyck_materials)


# pybind11
add_subdirectory(external/pybind11)

# Python module
file(GLOB PYCK_BINDINGS_SOURCES cpp/bindings/*.cpp)
pybind11_add_module(_pyck ${PYCK_BINDINGS_SOURCES})
target_link_libraries(_pyck PRIVATE pyck_basis pyck_geometry pyck_elements pyck_conditions pyck_constraints pyck_assembly pyck_materials pyck_postprocessing)
set_target_properties(_pyck PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/pyck")
install(TARGETS _pyck DESTINATION pyck)

# Test executable (unified runner)
file(GLOB_RECURSE TEST_SOURCES tests/cpp/*/test_*.cpp)
add_executable(pyck_tests
    tests/cpp/test_main.cpp
    tests/cpp/support/factories.cpp
    ${TEST_SOURCES}
)
target_include_directories(pyck_tests PRIVATE tests/cpp/support)
target_link_libraries(pyck_tests PRIVATE
    pyck_basis
    pyck_geometry
    pyck_operators
    pyck_elements
    pyck_conditions
    pyck_constraints
    pyck_assembly
    pyck_materials
    pyck_quadrature
    pyck_postprocessing
    catch
)

enable_testing()
add_test(NAME basis_tests COMMAND pyck_tests [basis])
add_test(NAME geometry_tests COMMAND pyck_tests [geometry])
add_test(NAME quadrature_tests COMMAND pyck_tests [quadrature])
add_test(NAME assembly_tests COMMAND pyck_tests [assembly])
add_test(NAME element_tests COMMAND pyck_tests [elements])
add_test(NAME material_tests COMMAND pyck_tests [materials])
