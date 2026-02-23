#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "tensor.hpp"
#include "surface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    py::class_<pyck::Basis, std::shared_ptr<pyck::Basis>>(m, "Basis")
        .def("degree", &pyck::Basis::degree)
        .def("num_basis", &pyck::Basis::num_basis)
        .def("eval", &pyck::Basis::eval_derivs,
             py::arg("u"), py::arg("order") = 0);

    py::class_<pyck::BSpline, pyck::Basis, std::shared_ptr<pyck::BSpline>>(m, "BSpline")
        .def(py::init<std::size_t, const std::vector<double>&>(),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &pyck::BSpline::knots);

    py::class_<pyck::TensorProduct>(m, "TensorProduct")
        .def(py::init<std::vector<std::shared_ptr<pyck::Basis>>>(),
             py::arg("bases"))
        .def("dim", &pyck::TensorProduct::dim)
        .def("num_basis", &pyck::TensorProduct::num_basis)
        .def("eval", &pyck::TensorProduct::eval,
             py::arg("params"))
        .def("eval_derivs", &pyck::TensorProduct::eval_derivs,
             py::arg("params"), py::arg("orders"));

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
