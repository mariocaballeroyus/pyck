#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "material.hpp"
#include "uniaxial_stress_1d.hpp"
#include "plane_stress_2d.hpp"

namespace py = pybind11;

namespace pyck {

void bind_materials(py::module_& m)
{

    // === Material1d =================================================================

    using Material1d = Material<double, 1>;

    py::class_<Material1d, Ptr<Material1d>>(m, "Material1d")
        // Abstract base class (no constructor)

        // Public attributes
        .def("shear_modulus", &Material1d::shear_modulus)
        .def("bending_stiffness", &Material1d::bending_stiffness)
        .def("shear_stiffness", &Material1d::shear_stiffness);

    py::class_<UniaxialStress1d<double>, Material1d, Ptr<UniaxialStress1d<double>>>(m, "UniaxialStress1d")
        // Constructor
        .def(py::init<double, double, double, double, double, double, double, double, double>(),
             py::arg("E"), py::arg("nu"), py::arg("A"), py::arg("I22"),
             py::arg("k22") = 5.0 / 6.0,
             py::arg("rho") = 0.0,
             py::arg("I33") = 0.0,
             py::arg("k33") = 0.0,
             py::arg("J")   = 0.0);

    // === Material2d =================================================================

    using Material2d = Material<double, 2>;

    py::class_<Material2d, Ptr<Material2d>>(m, "Material2d")
        // Abstract base class (no constructor)

        // Public attributes
        .def("bending_stiffness", &Material2d::bending_stiffness)
        .def("shear_stiffness", &Material2d::shear_stiffness);

    py::class_<PlaneStress2d<double>, Material2d, Ptr<PlaneStress2d<double>>>(m, "PlaneStress2d")
        // Constructor
        .def(py::init<double, double, double, double, double>(),
             py::arg("E"), py::arg("nu"), py::arg("t"),
             py::arg("k") = 5.0 / 6.0,
             py::arg("rho") = 0.0);
}

}
