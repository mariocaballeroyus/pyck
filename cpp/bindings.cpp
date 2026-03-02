#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "tensor.hpp"
#include "curve.hpp"
#include "dof_mapper.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "element.hpp"
#include "euler_bernoulli_beam_1p.hpp"
#include "condition.hpp"
#include "load_condition.hpp"
#include "assign_scalar.hpp"
#include "linear_elastic_problem.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    using BasisD = pyck::Basis<double>;
    py::class_<BasisD, pyck::Ptr<BasisD>>(m, "Basis")
        .def("degree", &BasisD::degree)
        .def("num_basis", &BasisD::num_basis)
        .def("find_span", &BasisD::find_span,
             py::arg("u"))
        .def("eval", &BasisD::eval_derivs,
             py::arg("u"), py::arg("span"), py::arg("order") = 0);

    using KnotVectorD = pyck::KnotVector<double>;
    py::class_<KnotVectorD>(m, "KnotVector")
        .def(py::init<std::vector<double>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &pyck::clamped_uniform_knots<double>,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &KnotVectorD::size)
        .def("num_basis", &KnotVectorD::num_basis,
             py::arg("degree"))
        .def("find_span", &KnotVectorD::find_span,
             py::arg("degree"), py::arg("u"))
        .def("num_spans", &KnotVectorD::num_spans)
        .def("span_bounds", &KnotVectorD::span_bounds,
             py::arg("span"))
        .def("data", &KnotVectorD::data)
        .def("__getitem__", [](const KnotVectorD& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorD::size);

    using BSplineD = pyck::BSpline<double>;
    py::class_<BSplineD, BasisD, pyck::Ptr<BSplineD>>(m, "BSpline")
        .def(py::init([](std::size_t degree, const std::vector<double>& knots) {
                 return std::make_shared<BSplineD>(degree, KnotVectorD(knots));
             }),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &BSplineD::knots);

    using Patch3D1D = pyck::Patch<double, 1>;
    py::class_<Patch3D1D, pyck::Ptr<Patch3D1D>>(m, "Patch3D1D")
        .def("gdim", &Patch3D1D::gdim)
        .def("tdim", &Patch3D1D::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<double, 3>& (Patch3D1D::*)() const>(
                 &Patch3D1D::control_pts),
             py::return_value_policy::reference_internal)
        .def("num_control_pts", &Patch3D1D::num_control_pts)
        .def("eval_jacobian", &Patch3D1D::eval_jacobian,
             py::arg("params"), py::arg("spans"))
        .def("boundary_dofs", &Patch3D1D::boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("dof_mapper", &Patch3D1D::dof_mapper,
             py::return_value_policy::reference_internal)
        .def("active_control_pts", &Patch3D1D::active_control_pts,
             py::arg("spans"));

    using CurvePatch3D = pyck::CurvePatch<double>;
    py::class_<CurvePatch3D, Patch3D1D, pyck::Ptr<CurvePatch3D>>(m, "CurvePatch")
        .def(py::init([](pyck::Ptr<BSplineD> basis,
                         const pyck::ColMatrix<double, 3>& cp) {
                 return std::make_shared<CurvePatch3D>(std::static_pointer_cast<BasisD>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3D::eval_basis_functions,
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const CurvePatch3D& p,
                 const pyck::ColMatrix<double, 1>& pts,
                 const std::array<pyck::Index, 1>& spans,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, spans, order);
             },
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3D::eval_geometry,
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_jacobian", &CurvePatch3D::eval_jacobian,
             py::arg("params"), py::arg("spans"));

    // line_segment factory
    m.def("line_segment", [](pyck::Ptr<BSplineD> basis, double length) {
            return std::make_shared<CurvePatch3D>(
                pyck::line_segment<double>(
                    std::static_pointer_cast<const BasisD>(basis), length));
        },
        py::arg("basis"), py::arg("length"),
        "Create a straight line segment C(u) = (L*u, 0, 0) along the x-axis.");

    // === DOF Mapping ================================================================

    using DofMapper1D = pyck::DofMapper<1>;
    py::class_<DofMapper1D>(m, "DofMapper1D")
        .def("get_boundary_dofs", &DofMapper1D::get_boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("get_element_dofs",
             py::overload_cast<pyck::Index>(&DofMapper1D::get_element_dofs, py::const_),
             py::arg("elem_idx"))
        .def("num_basis", &DofMapper1D::num_basis)
        .def("degree", &DofMapper1D::degree);

    // === Quadrature =================================================================

    using QR1D = pyck::QuadratureRule<double, 1>;
    py::class_<QR1D, pyck::Ptr<QR1D>>(m, "QuadratureRule1D")
        .def("points", &QR1D::points, py::return_value_policy::reference_internal)
        .def("weights", &QR1D::weights, py::return_value_policy::reference_internal)
        .def("num_points", &QR1D::num_points);

    using GL1D = pyck::GaussLegendre<double, 1>;
    py::class_<GL1D, QR1D, pyck::Ptr<GL1D>>(m, "GaussLegendre1D")
        .def(py::init<std::size_t>(), py::arg("num_pts"));

    // === Elements ===================================================================

    using Elem1D = pyck::Element<double, 1>;
    py::class_<Elem1D, pyck::Ptr<Elem1D>>(m, "Element1D");

    using EBB = pyck::EulerBernoulliBeam1P<double>;
    py::class_<EBB, Elem1D, pyck::Ptr<EBB>>(m, "EulerBernoulliBeam1P")
        .def(py::init<double, double, double>(),
             py::arg("E"), py::arg("A"), py::arg("I"));

    // === Conditions =================================================================

    using CondD = pyck::Condition<double>;
    py::class_<CondD, pyck::Ptr<CondD>>(m, "Condition");

    using LC1D = pyck::LoadCondition<double, 1>;
    py::class_<LC1D, CondD, pyck::Ptr<LC1D>>(m, "LoadCondition1D")
        .def(py::init<const Patch3D1D&, const QR1D&, const pyck::Vector<double>&>(),
             py::arg("patch"), py::arg("quadrature"), py::arg("load_values"));

    m.def("assign_scalar",
        [](const std::vector<pyck::Index>& dofs,
           const std::vector<double>& values,
           pyck::Matrix<double> K,
           pyck::Vector<double> F) {
            pyck::assign_scalar<double>(dofs, values, K, F);
            return py::make_tuple(std::move(K), std::move(F));
        },
        py::arg("dofs"), py::arg("values"),
        py::arg("stiffness"), py::arg("load"),
        "Apply Dirichlet BCs with prescribed values. Returns (K, F).");

    m.def("assign_zeros",
        [](const std::vector<pyck::Index>& dofs,
           pyck::Matrix<double> K,
           pyck::Vector<double> F) {
            pyck::assign_zeros<double>(dofs, K, F);
            return py::make_tuple(std::move(K), std::move(F));
        },
        py::arg("dofs"), py::arg("stiffness"), py::arg("load"),
        "Apply homogeneous Dirichlet BCs. Returns (K, F).");

    // === Assembly ===================================================================

    using LEP1D = pyck::LinearElasticProblem<double, 1>;
    py::class_<LEP1D>(m, "LinearElasticProblem1D")
        .def(py::init<const pyck::Ptr<Patch3D1D>&,
                      const pyck::Ptr<Elem1D>&,
                      const pyck::Ptr<QR1D>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"))
        .def("add_condition", &LEP1D::add_condition,
             py::arg("condition"))
        .def("assemble", [](const LEP1D& p) {
            pyck::Matrix<double> K;
            pyck::Vector<double> F;
            p.assemble(K, F);
            return py::make_tuple(std::move(K), std::move(F));
        }, "Assemble global stiffness matrix and load vector. Returns (K, F).");

#ifdef PYCK_BUILD_SINGLE_PRECISION
    using BasisF = pyck::Basis<float>;
    py::class_<BasisF, pyck::Ptr<BasisF>>(m, "Basis32")
        .def("degree", &BasisF::degree)
        .def("num_basis", &BasisF::num_basis)
        .def("find_span", &BasisF::find_span, py::arg("u"))
        .def("eval", &BasisF::eval_derivs,
             py::arg("u"), py::arg("span"), py::arg("order") = 0);

    using KnotVectorF = pyck::KnotVector<float>;
    py::class_<KnotVectorF>(m, "KnotVector32")
        .def(py::init<std::vector<float>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &pyck::clamped_uniform_knots<float>,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &KnotVectorF::size)
        .def("num_spans", &KnotVectorF::num_spans)
        .def("num_basis", &KnotVectorF::num_basis,
             py::arg("degree"))
        .def("find_span", &KnotVectorF::find_span,
             py::arg("degree"), py::arg("u"))
        .def("span_bounds", &KnotVectorF::span_bounds,
             py::arg("span"))
        .def("data", &KnotVectorF::data)
        .def("__getitem__", [](const KnotVectorF& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorF::size);

    using BSplineF = pyck::BSpline<float>;
    py::class_<BSplineF, BasisF, pyck::Ptr<BSplineF>>(m, "BSpline32")
        .def(py::init([](std::size_t degree, const std::vector<float>& knots) {
                 return std::make_shared<BSplineF>(degree, KnotVectorF(knots));
             }),
             py::arg("degree"), py::arg("knots"))
        .def("knots", &BSplineF::knots);

    using Patch3D1DF = pyck::Patch<float, 1>;
    py::class_<Patch3D1DF, pyck::Ptr<Patch3D1DF>>(m, "Patch3D1D32")
        .def("gdim", &Patch3D1DF::gdim)
        .def("tdim", &Patch3D1DF::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<float, 3>& (Patch3D1DF::*)() const>(
                 &Patch3D1DF::control_pts),
             py::return_value_policy::reference_internal)
        .def("num_control_pts", &Patch3D1DF::num_control_pts)
        .def("eval_jacobian", &Patch3D1DF::eval_jacobian,
             py::arg("params"), py::arg("spans"))
        .def("boundary_dofs", &Patch3D1DF::boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("dof_mapper", &Patch3D1DF::dof_mapper,
             py::return_value_policy::reference_internal);

    using CurvePatch3DF = pyck::CurvePatch<float>;
    py::class_<CurvePatch3DF, Patch3D1DF, pyck::Ptr<CurvePatch3DF>>(m, "CurvePatch32")
        .def(py::init([](pyck::Ptr<BSplineF> basis,
                         const pyck::ColMatrix<float, 3>& cp) {
                 return new CurvePatch3DF(std::static_pointer_cast<BasisF>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3DF::eval_basis_functions,
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const CurvePatch3DF& p,
                 const pyck::ColMatrix<float, 1>& pts,
                 const std::array<pyck::Index, 1>& spans,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, spans, order);
             },
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3DF::eval_geometry,
             py::arg("params"), py::arg("spans"), py::arg("order") = 0)
        .def("eval_jacobian", &CurvePatch3DF::eval_jacobian,
             py::arg("params"), py::arg("spans"));
#endif
}
