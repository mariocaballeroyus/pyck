#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "quadrature.hpp"
#include "gauss_legendre.hpp"

namespace py = pybind11;

namespace pyck {

void bind_quadrature(py::module_& m)
{

    // === QuadratureRule =============================================================

    using QuadratureRule1d = QuadratureRule<double, 1>;

    py::class_<QuadratureRule1d, Ptr<QuadratureRule1d>>(m, "QuadratureRule1d")
        // Abstract base class (no constructor)

        // Zero-copy view over quadrature points
        .def("points", &QuadratureRule1d::points, 
             py::return_value_policy::reference_internal)

        // Zero-copy view over weights
        .def("weights", &QuadratureRule1d::weights,
             py::return_value_policy::reference_internal);

    // === Higher-dimensional QuadratureRule =========================================

    using QuadratureRule2d = QuadratureRule<double, 2>;

    py::class_<QuadratureRule2d, Ptr<QuadratureRule2d>>(m, "QuadratureRule2d")
        // Abstract base class (no constructor)

        // Constructors
        .def(py::init<const QuadratureRule1d&>(), 
             py::arg("rule"))
        .def(py::init<std::array<const QuadratureRule1d*, 2>>(), 
             py::arg("rules"))

        // Zero-copy view over quadrature points
        .def("points", &QuadratureRule2d::points, 
             py::return_value_policy::reference_internal)

        // Zero-copy view over weights
        .def("weights", &QuadratureRule2d::weights,
             py::return_value_policy::reference_internal);

    // === Gauss-Legendre(QuadratureRule) =============================================

    py::class_<GaussLegendre<double, 1>, QuadratureRule1d, Ptr<GaussLegendre<double, 1>>>(m, "GaussLegendre1d")
        // Constructor
        .def(py::init<std::size_t>(), 
             py::arg("num_pts"));

    py::class_<GaussLegendre<double, 2>, QuadratureRule2d, Ptr<GaussLegendre<double, 2>>>(m, "GaussLegendre2d")
        // Constructor
        .def(py::init<std::size_t>(), 
             py::arg("num_pts"));

}

}
