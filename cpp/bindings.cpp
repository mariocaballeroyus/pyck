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

    py::class_<pyck::KnotVector>(m, "KnotVector")
        .def(py::init<std::vector<double>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &pyck::KnotVector::clamped_uniform,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &pyck::KnotVector::size)
        .def("num_basis", &pyck::KnotVector::num_basis,
             py::arg("degree"))
        .def("find_span", &pyck::KnotVector::find_span,
             py::arg("degree"), py::arg("u"))
        .def("data", &pyck::KnotVector::data)
        .def("__getitem__", [](const pyck::KnotVector& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &pyck::KnotVector::size);

    py::class_<pyck::BSpline, pyck::Basis, std::shared_ptr<pyck::BSpline>>(m, "BSpline")
        .def(py::init([](std::size_t degree, const std::vector<double>& knots) {
                 return std::make_shared<pyck::BSpline>(degree, pyck::KnotVector(knots));
             }),
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
             py::arg("params"))
        .def("jacobian_det", &pyck::Patch::jacobian_det,
             py::arg("params"));

    py::class_<pyck::SurfacePatch, pyck::Patch>(m, "SurfacePatch")
        .def(py::init([](std::size_t gdim,
                         std::array<pyck::Basis*, 2> bases,
                         const Eigen::MatrixXd& cp) {
                 return new pyck::SurfacePatch(gdim, bases, cp);
             }),
             py::arg("gdim"), py::arg("bases"),
             py::arg("control_points"),
             py::keep_alive<1, 3>())
        .def("eval", &pyck::SurfacePatch::eval,
             py::arg("params"))
        .def("eval_physical_derivs", &pyck::SurfacePatch::eval_physical_derivs,
             py::arg("params"), py::arg("order") = 1)
        .def("jacobian", &pyck::SurfacePatch::jacobian,
             py::arg("params"))
        .def("jacobian_det", &pyck::SurfacePatch::jacobian_det,
             py::arg("params"));
}
