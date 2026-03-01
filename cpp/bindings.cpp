#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "tensor.hpp"
#include "curve.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    using BasisD = pyck::Basis<double>;
    py::class_<BasisD, std::shared_ptr<BasisD>>(m, "Basis")
        .def("degree", &BasisD::degree)
        .def("num_basis", &BasisD::num_basis)
        .def("eval", &BasisD::eval_derivs,
             py::arg("u"), py::arg("order") = 0);

    using KnotVectorD = pyck::KnotVector<double>;
    py::class_<KnotVectorD>(m, "KnotVector")
        .def(py::init<std::vector<double>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &KnotVectorD::clamped_uniform,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &KnotVectorD::size)
        .def("num_basis", &KnotVectorD::num_basis,
             py::arg("degree"))
        .def("find_span", &KnotVectorD::find_span,
             py::arg("degree"), py::arg("u"))
        .def("data", &KnotVectorD::data)
        .def("__getitem__", [](const KnotVectorD& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorD::size);

    using BSplineD = pyck::BSpline<double>;
    py::class_<BSplineD, BasisD, std::shared_ptr<BSplineD>>(m, "BSpline")
        .def(py::init([](std::size_t degree, const std::vector<double>& knots) {
                 return std::make_shared<BSplineD>(degree, KnotVectorD(knots));
             }),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &BSplineD::knots);

    using Patch3D1D = pyck::Patch<double, 1>;
    py::class_<Patch3D1D>(m, "Patch3D1D")
        .def("gdim", &Patch3D1D::gdim)
        .def("tdim", &Patch3D1D::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<double, 3>& (Patch3D1D::*)() const>(
                 &Patch3D1D::control_pts),
             py::return_value_policy::reference_internal)
        .def("eval_jacobian", &Patch3D1D::eval_jacobian,
             py::arg("params"));

    using CurvePatch3D = pyck::CurvePatch<double>;
    py::class_<CurvePatch3D, Patch3D1D>(m, "CurvePatch")
        .def(py::init([](std::shared_ptr<BSplineD> basis,
                         const pyck::ColMatrix<double, 3>& cp) {
                 return new CurvePatch3D(std::static_pointer_cast<BasisD>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3D::eval_basis_functions,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_shape_functions", &CurvePatch3D::eval_shape_functions,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3D::eval_geometry,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_jacobian", &CurvePatch3D::eval_jacobian,
             py::arg("params"));

#ifdef PYCK_BUILD_SINGLE_PRECISION
    using BasisF = pyck::Basis<float>;
    py::class_<BasisF, std::shared_ptr<BasisF>>(m, "Basis32")
        .def("degree", &BasisF::degree)
        .def("num_basis", &BasisF::num_basis)
        .def("eval", &BasisF::eval_derivs,
             py::arg("u"), py::arg("order") = 0);

    using KnotVectorF = pyck::KnotVector<float>;
    py::class_<KnotVectorF>(m, "KnotVector32")
        .def(py::init<std::vector<float>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &KnotVectorF::clamped_uniform,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &KnotVectorF::size)
        .def("num_basis", &KnotVectorF::num_basis,
             py::arg("degree"))
        .def("find_span", &KnotVectorF::find_span,
             py::arg("degree"), py::arg("u"))
        .def("data", &KnotVectorF::data)
        .def("__getitem__", [](const KnotVectorF& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorF::size);

    using BSplineF = pyck::BSpline<float>;
    py::class_<BSplineF, BasisF, std::shared_ptr<BSplineF>>(m, "BSpline32")
        .def(py::init([](std::size_t degree, const std::vector<float>& knots) {
                 return std::make_shared<BSplineF>(degree, KnotVectorF(knots));
             }),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &BSplineF::knots);

    using Patch3D1DF = pyck::Patch<float, 1>;
    py::class_<Patch3D1DF>(m, "Patch3D1D32")
        .def("gdim", &Patch3D1DF::gdim)
        .def("tdim", &Patch3D1DF::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<float, 3>& (Patch3D1DF::*)() const>(
                 &Patch3D1DF::control_pts),
             py::return_value_policy::reference_internal)
        .def("eval_jacobian", &Patch3D1DF::eval_jacobian,
             py::arg("params"));

    using CurvePatch3DF = pyck::CurvePatch<float>;
    py::class_<CurvePatch3DF, Patch3D1DF>(m, "CurvePatch32")
        .def(py::init([](std::shared_ptr<BSplineF> basis,
                         const pyck::ColMatrix<float, 3>& cp) {
                 return new CurvePatch3DF(std::static_pointer_cast<BasisF>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3DF::eval_basis_functions,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_shape_functions", &CurvePatch3DF::eval_shape_functions,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3DF::eval_geometry,
             py::arg("params"), py::arg("order") = 0)
        .def("eval_jacobian", &CurvePatch3DF::eval_jacobian,
             py::arg("params"));
#endif
}
