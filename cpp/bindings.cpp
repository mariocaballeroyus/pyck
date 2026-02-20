#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "surface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    py::class_<pyck::Basis>(m, "Basis")
        .def("degree", &pyck::Basis::degree)
        .def("num_basis", &pyck::Basis::num_basis)
        .def("eval", &pyck::Basis::eval,
             py::arg("u"), py::arg("order") = 0);

    py::class_<pyck::BSpline, pyck::Basis>(m, "BSpline")
        .def(py::init<std::size_t, const std::vector<double>&>(),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &pyck::BSpline::knots);

    py::class_<pyck::Patch>(m, "Patch")
        .def("gdim", &pyck::Patch::gdim)
        .def("tdim", &pyck::Patch::tdim)
        .def("control_points",
             static_cast<const Eigen::MatrixXd& (pyck::Patch::*)() const>(
                 &pyck::Patch::control_points),
             py::return_value_policy::reference_internal)
        .def("jacobian", &pyck::Patch::jacobian,
             py::arg("u"), py::arg("v"));

    py::class_<pyck::SurfacePatch, pyck::Patch>(m, "SurfacePatch")
        .def(py::init([](std::size_t gdim,
                         pyck::BSpline* basis_u,
                         pyck::BSpline* basis_v,
                         const Eigen::MatrixXd& cp) {
                 return new pyck::SurfacePatch(gdim, basis_u, basis_v, cp);
             }),
             py::arg("gdim"), py::arg("basis_u"),
             py::arg("basis_v"), py::arg("control_points"),
             py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("eval", &pyck::SurfacePatch::eval,
             py::arg("u"), py::arg("v"), py::arg("order") = 0)
        .def("jacobian", &pyck::SurfacePatch::jacobian,
             py::arg("u"), py::arg("v"))
        .def("tensor_basis", &pyck::SurfacePatch::tensor_basis,
             py::arg("u"), py::arg("v"), py::arg("order") = 0);
}
