#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "tensor.hpp"
#include "curve.hpp"
#include "boundary_patch.hpp"
#include "dof_mapper.hpp"
#include "quadrature.hpp"
#include "quadrature_utils.hpp"
#include "gauss_legendre.hpp"
#include "element.hpp"
#include "euler_bernoulli_beam_1p.hpp"
#include "timoshenko_beam_2p.hpp"
#include "condition.hpp"
#include "load_condition.hpp"
#include "dirichlet_bc.hpp"
#include "constraint.hpp"
#include "master_slave_constraint.hpp"
#include "linear_elastic_problem.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    using BasisD = pyck::Basis<double>;
    py::class_<BasisD, pyck::Ptr<BasisD>>(m, "Basis")
        .def("degree", &BasisD::degree)
        .def("num_basis", &BasisD::num_basis)
        .def("eval", &BasisD::eval_on_span,
             py::arg("u"), py::arg("span"), py::arg("order") = 0)
        .def("eval_all", &BasisD::eval_all,
             py::arg("u"), py::arg("order") = 0,
             "Evaluate all basis functions and derivatives (vector of matrices).");

    using KnotVectorD = pyck::KnotVector<double>;
    py::class_<KnotVectorD>(m, "KnotVector")
        .def(py::init<std::vector<double>>(),
             py::arg("knots"))
        .def_static("clamped_uniform", &pyck::clamped_uniform_knots<double>,
             py::arg("degree"), py::arg("num_basis"))
        .def("size", &KnotVectorD::size)
        .def("num_basis", &KnotVectorD::num_basis,
             py::arg("degree"))
        .def("num_spans", &KnotVectorD::num_spans)
        .def("span_bounds", &KnotVectorD::span_bounds,
             py::arg("span"))
        .def("find_span", &KnotVectorD::find_span,
             py::arg("degree"), py::arg("u"))
        .def("data", [](KnotVectorD& kv) {
             return py::array_t<double>(
                 kv.size(),
                 kv.data().data(),
                 py::cast(&kv));
         })
        .def("__getitem__", [](const KnotVectorD& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorD::size);

    using BSplineD = pyck::BSpline<double>;
    py::class_<BSplineD, BasisD, pyck::Ptr<BSplineD>>(m, "BSpline")
        .def(py::init<std::size_t, KnotVectorD>(),
             py::arg("degree"), py::arg("knot_vector"))
        .def("knots", [](BSplineD& bs) {
             const auto& v = bs.knots();
             return py::array_t<double>(
                 v.size(),
                 v.data(),
                 py::cast(&bs));
         });

    // === BoundaryPatch (1D parent) ================================================

    using BoundaryPatch1D = pyck::BoundaryPatch<double, 1>;
    py::class_<BoundaryPatch1D, pyck::Ptr<BoundaryPatch1D>>(m, "BoundaryPatch1D")
        .def("boundary_dofs", &BoundaryPatch1D::boundary_dofs)
        .def("displacement_dofs", &BoundaryPatch1D::displacement_dofs)
        .def("rotation_dofs", &BoundaryPatch1D::rotation_dofs)
        .def("param_dim", &BoundaryPatch1D::param_dim)
        .def("at_start", &BoundaryPatch1D::at_start);

    using Patch3D1D = pyck::Patch<double, 1>;
    py::class_<Patch3D1D, pyck::Ptr<Patch3D1D>>(m, "Patch3D1D")
        .def("gdim", &Patch3D1D::gdim)
        .def("tdim", &Patch3D1D::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<double, 3>& (Patch3D1D::*)() const>(
                 &Patch3D1D::control_pts),
             py::return_value_policy::reference_internal)
        .def("num_control_pts", &Patch3D1D::num_control_pts)
        .def("boundary_dofs", &Patch3D1D::boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("dof_mapper", &Patch3D1D::dof_mapper,
             py::return_value_policy::reference_internal)
        .def("active_control_pts", &Patch3D1D::active_control_pts,
             py::arg("spans"))
        .def("boundary", &Patch3D1D::boundary,
             py::arg("param_dim"), py::arg("at_start"),
             "Extract a boundary face of this patch.");

    using CurvePatch3D = pyck::CurvePatch<double>;
    py::class_<CurvePatch3D, Patch3D1D, pyck::Ptr<CurvePatch3D>>(m, "CurvePatch")
        .def(py::init([](pyck::Ptr<BSplineD> basis,
                         const pyck::ColMatrix<double, 3>& cp) {
                 return std::make_shared<CurvePatch3D>(std::static_pointer_cast<BasisD>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3D::eval_basis_functions,
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const CurvePatch3D& p,
                 const pyck::ColMatrix<double, 1>& pts,
                 pyck::Index span,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, span, order);
             },
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3D::eval_geometry,
             py::arg("params"), py::arg("span"));

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
        .def("get_displacement_boundary_dofs", &DofMapper1D::get_displacement_boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("get_rotation_boundary_dofs", &DofMapper1D::get_rotation_boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("get_element_dofs",
             py::overload_cast<pyck::Index>(&DofMapper1D::get_element_dofs, py::const_),
             py::arg("elem_idx"))
        .def("num_basis", &DofMapper1D::num_basis)
        .def("degree", &DofMapper1D::degree);

    // === Quadrature =================================================================

    using QR = pyck::QuadratureRule<double>;
    py::class_<QR, pyck::Ptr<QR>>(m, "QuadratureRule")
        .def("points", &QR::points, py::return_value_policy::reference_internal)
        .def("weights", &QR::weights, py::return_value_policy::reference_internal)
        .def("num_points", &QR::num_points)
        .def("map_to_domain", &QR::map_to_domain,
             py::arg("lo"), py::arg("hi"));

    using GL = pyck::GaussLegendre<double>;
    py::class_<GL, QR, pyck::Ptr<GL>>(m, "GaussLegendre")
        .def(py::init<std::size_t>(), py::arg("num_pts"));

    // Tensor-product utilities exposed to Python
    m.def("tensor_product_1d",
        [](const QR& rule) {
            std::array<const QR*, 1> rules = {&rule};
            return pyck::tensor_product<double, 1>(rules);
        },
        py::arg("rule"),
        "Return (points, weights) for a 1D tensor product (identity).");

    m.def("tensor_product_2d",
        [](const QR& rule_u, const QR& rule_v) {
            std::array<const QR*, 2> rules = {&rule_u, &rule_v};
            return pyck::tensor_product<double, 2>(rules);
        },
        py::arg("rule_u"), py::arg("rule_v"),
        "Form the 2D tensor product of two 1D quadrature rules.");

    m.def("tensor_product_3d",
        [](const QR& rule_u, const QR& rule_v, const QR& rule_w) {
            std::array<const QR*, 3> rules = {&rule_u, &rule_v, &rule_w};
            return pyck::tensor_product<double, 3>(rules);
        },
        py::arg("rule_u"), py::arg("rule_v"), py::arg("rule_w"),
        "Form the 3D tensor product of three 1D quadrature rules.");

    // === Elements ===================================================================

    using Elem1D = pyck::Element<double, 1>;
    py::class_<Elem1D, pyck::Ptr<Elem1D>>(m, "Element1D")
        .def("num_dofs_per_node", &Elem1D::num_dofs_per_node);

    using EBB = pyck::EulerBernoulliBeam1P<double>;
    py::class_<EBB, Elem1D, pyck::Ptr<EBB>>(m, "EulerBernoulliBeam1P")
        .def(py::init<double, double, double>(),
             py::arg("E"), py::arg("A"), py::arg("I"));

    using TBB = pyck::TimoshenkoBeam2P<double>;
    py::class_<TBB, Elem1D, pyck::Ptr<TBB>>(m, "TimoshenkoBeam2P")
        .def(py::init<double, double, double, double, double>(),
             py::arg("E"), py::arg("A"), py::arg("I"), py::arg("G"), py::arg("k") = 5.0 / 6.0);

    // === Conditions =================================================================

    using CondD = pyck::Condition<double>;
    py::class_<CondD, pyck::Ptr<CondD>>(m, "Condition");

    using LC1D = pyck::LoadCondition<double, 1>;
    py::class_<LC1D, CondD, pyck::Ptr<LC1D>>(m, "LoadCondition1D")
        .def(py::init<const Patch3D1D&, const Elem1D&, const QR&, const pyck::Vector<double>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"), py::arg("load_values"));

    // === Constraints ================================================================

    using ConstD = pyck::Constraint<double>;
    py::class_<ConstD, pyck::Ptr<ConstD>>(m, "Constraint");

    using DBC = pyck::DirichletBC<double>;
    py::class_<DBC, pyck::Ptr<DBC>>(m, "DirichletBC")
        .def(py::init<std::vector<pyck::Index>, std::vector<double>>(),
             py::arg("dofs"), py::arg("values"))
        .def(py::init<std::vector<pyck::Index>, double>(),
             py::arg("dofs"), py::arg("value") = 0.0)
        .def("dofs", &DBC::dofs)
        .def("values", &DBC::values)
        .def("apply", &DBC::apply, py::arg("stiffness"), py::arg("load"));

    using MSC = pyck::MasterSlaveConstraint<double>;
    py::class_<MSC, ConstD, pyck::Ptr<MSC>>(m, "MasterSlaveConstraint")
        .def(py::init<std::vector<std::pair<pyck::Index, pyck::Index>>>(),
             py::arg("master_slave_pairs"))
        .def("pairs", &MSC::pairs)
        .def("apply", &MSC::apply, py::arg("stiffness"), py::arg("load"));



    // === Assembly ===================================================================

    // --- Quadrature utilities ---
    m.def("eval_physical_quadrature_points",
        [](const pyck::CurvePatch<double>& patch,
           const pyck::QuadratureRule<double>& quad) {
            return pyck::eval_physical_quadrature_points<double>(patch, quad);
        },
        py::arg("patch"), py::arg("quadrature"),
        "Physical x-coordinates of all active quadrature points "
        "(assembly layer, correct dependency direction).");

    using LEP1D = pyck::LinearElasticProblem<double, 1>;
    py::class_<LEP1D>(m, "LinearElasticProblem1D")
        .def(py::init<const pyck::Ptr<Patch3D1D>&,
                      const pyck::Ptr<Elem1D>&>(),
             py::arg("patch"), py::arg("element"))
        .def("set_quadrature", &LEP1D::set_quadrature,
             py::arg("quadrature"))
        .def("add_condition", &LEP1D::add_condition,
             py::arg("condition"))
        .def("add_constraint", &LEP1D::add_constraint,
             py::arg("constraint"))
        .def("add_constraint", &LEP1D::add_dirichlet_bc,
             py::arg("dirichlet_bc"))
        .def("add_dirichlet_bc", &LEP1D::add_dirichlet_bc,
             py::arg("bc"))
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
        .def("eval", &BasisF::eval_on_span,
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
        .def("data", [](KnotVectorF& kv) {
             return py::array_t<float>(
                 kv.size(),
                 kv.data().data(),
                 py::cast(&kv));
         })
        .def("__getitem__", [](const KnotVectorF& kv, std::size_t i) {
             return kv[i];
         })
        .def("__len__", &KnotVectorF::size);

    using BSplineF = pyck::BSpline<float>;
    py::class_<BSplineF, BasisF, pyck::Ptr<BSplineF>>(m, "BSpline32")
        .def(py::init<std::size_t, KnotVectorF>(),
             py::arg("degree"), py::arg("knot_vector"))
        .def("knots", [](BSplineF& bs) {
             const auto& v = bs.knots();
             return py::array_t<float>(
                 v.size(),
                 v.data(),
                 py::cast(&bs));
         });

    using Patch3D1DF = pyck::Patch<float, 1>;
    py::class_<Patch3D1DF, pyck::Ptr<Patch3D1DF>>(m, "Patch3D1D32")
        .def("gdim", &Patch3D1DF::gdim)
        .def("tdim", &Patch3D1DF::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<float, 3>& (Patch3D1DF::*)() const>(
                 &Patch3D1DF::control_pts),
             py::return_value_policy::reference_internal)
        .def("num_control_pts", &Patch3D1DF::num_control_pts)
        .def("boundary_dofs", &Patch3D1DF::boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("dof_mapper", &Patch3D1DF::dof_mapper,
             py::return_value_policy::reference_internal)
        .def("boundary", &Patch3D1DF::boundary,
             py::arg("param_dim"), py::arg("at_start"),
             "Extract a boundary face of this patch.");

    using CurvePatch3DF = pyck::CurvePatch<float>;
    py::class_<CurvePatch3DF, Patch3D1DF, pyck::Ptr<CurvePatch3DF>>(m, "CurvePatch32")
        .def(py::init([](pyck::Ptr<BSplineF> basis,
                         const pyck::ColMatrix<float, 3>& cp) {
                 return new CurvePatch3DF(std::static_pointer_cast<BasisF>(basis), cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
        .def("eval_basis_functions", &CurvePatch3DF::eval_basis_functions,
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const CurvePatch3DF& p,
                 const pyck::ColMatrix<float, 1>& pts,
                 pyck::Index span,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, span, order);
             },
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_geometry", &CurvePatch3DF::eval_geometry,
             py::arg("params"), py::arg("span"));
#endif
}
