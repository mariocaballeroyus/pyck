#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "tensor.hpp"
#include "factories.hpp"
#include "factories.hpp"
#include "boundary_patch.hpp"
#include "dof_mapper.hpp"
#include "quadrature.hpp"
#include "gauss_legendre.hpp"
#include "element.hpp"
#include "beam_euler_bernoulli_1p.hpp"
#include "condition.hpp"
#include "load_condition.hpp"
#include "linear_constraint.hpp"
#include "beam_timoshenko_1p.hpp"
#include "beam_timoshenko_2p.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "plate_reissner_mindlin_1p.hpp"
#include "linear_elastic_problem.hpp"
#include "material.hpp"
#include "slender_beam_1d.hpp"
#include "plane_stress_2d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pyck, m) {

    using BasisD = pyck::Basis<double>;
    py::class_<BasisD, pyck::Ptr<BasisD>>(m, "Basis")
        .def("degree", &BasisD::degree)
        .def("num_basis", &BasisD::num_basis)
        .def("num_intervals", &BasisD::num_intervals)
        .def("find_span", &BasisD::find_span, py::arg("u"))
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
        .def("span_bounds", &KnotVectorD::span_bounds,
             py::arg("span"))
        .def("find_span", &KnotVectorD::find_span,
             py::arg("degree"), py::arg("u"))
        .def("data", [](KnotVectorD& kv) {
             return py::array_t<double>(
                 kv.size(),
                 kv.data().data(),
                 py::cast(&kv));
         });

    using BSplineD = pyck::BSpline<double>;
    py::class_<BSplineD, BasisD, pyck::Ptr<BSplineD>>(m, "BSpline")
        .def(py::init<std::size_t, KnotVectorD>(),
             py::arg("degree"), py::arg("knot_vector"))
        .def("knot_vector", &BSplineD::knot_vector,
             py::return_value_policy::reference_internal);

    using Patch3D1D = pyck::Patch<double, 1>;
    py::class_<Patch3D1D, pyck::Ptr<Patch3D1D>>(m, "Patch1d")
        .def(py::init([](pyck::Ptr<BasisD> basis,
                         const pyck::ColMatrix<double, 3>& cp) {
                 return std::make_shared<Patch3D1D>(basis, cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
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
             py::return_value_policy::reference_internal,
             "Return the DofMapper of this patch.")
        .def("active_control_pts", &Patch3D1D::active_control_pts,
             py::arg("spans"))
        .def("get_control_points", &Patch3D1D::get_control_points,
             py::arg("indices"),
             "Map control point indices to physical coordinates.")
        .def("basis", &Patch3D1D::basis,
             py::arg("dir") = 0,
             py::return_value_policy::reference_internal)
        .def("eval_physical_points", [](const Patch3D1D& self, const pyck::QuadratureRule<double, 1>& quad) {
                 return self.eval_physical_points(quad);
             },
             py::arg("quadrature"),
             "Coordinate mapping of all quadrature points in active elements.")
        .def("eval_basis_functions", &Patch3D1D::eval_basis_functions,
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const Patch3D1D& p,
                 const pyck::ColMatrix<double, 1>& pts,
                 pyck::Index span,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, span, order);
             },
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_geometry", [](const Patch3D1D& self, const pyck::ColMatrix<double, 1>& pts, pyck::Index span) {
                 return self.eval_geometry(pts, span);
             },
             py::arg("params"), py::arg("span"));

    // line_segment factory
    m.def("line_segment", [](pyck::Ptr<BasisD> basis, double length) {
            return std::make_shared<Patch3D1D>(
                pyck::line_segment<double>(basis, length));
        },
        py::arg("basis"), py::arg("length"),
        "Create a straight line segment C(u) = (L*u, 0, 0) along the x-axis.");

    // === Surface Patch (2D) =========================================================

    using BoundaryPatch2D = pyck::BoundaryPatch<double, 2>;
    py::class_<BoundaryPatch2D, Patch3D1D, pyck::Ptr<BoundaryPatch2D>>(m, "BoundaryPatch2d")
        .def("parent_dofs", &BoundaryPatch2D::parent_dofs)
        .def("param_dim", &BoundaryPatch2D::param_dim)
        .def("at_start", &BoundaryPatch2D::at_start);

    using Patch3D2D = pyck::Patch<double, 2>;
    py::class_<Patch3D2D, pyck::Ptr<Patch3D2D>>(m, "Patch2d")
        .def(py::init([](pyck::Ptr<BasisD> basis_u,
                         pyck::Ptr<BasisD> basis_v,
                         const pyck::ColMatrix<double, 3>& cp) {
                 return std::make_shared<Patch3D2D>(
                     basis_u,
                     basis_v,
                     cp);
             }),
             py::arg("basis_u"),
             py::arg("basis_v"),
             py::arg("control_points"))
        .def("gdim", &Patch3D2D::gdim)
        .def("tdim", &Patch3D2D::tdim)
        .def("control_pts",
             static_cast<const pyck::ColMatrix<double, 3>& (Patch3D2D::*)() const>(
                 &Patch3D2D::control_pts),
             py::return_value_policy::reference_internal)
        .def("num_control_pts", &Patch3D2D::num_control_pts)
        .def("boundary_dofs", &Patch3D2D::boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("dof_mapper", &Patch3D2D::dof_mapper,
             py::return_value_policy::reference_internal,
             "Return the DofMapper of this patch.")
        .def("active_control_pts", &Patch3D2D::active_control_pts,
             py::arg("spans"))
        .def("get_control_points", &Patch3D2D::get_control_points,
             py::arg("indices"),
             "Map control point indices to physical coordinates.")
        .def("basis", &Patch3D2D::basis,
             py::arg("dir"),
             py::return_value_policy::reference_internal)
        .def("boundary", [](pyck::Ptr<Patch3D2D> self, std::size_t param_dim, bool at_start) {
                 return pyck::create_boundary<double, 2>(self, param_dim, at_start);
             },
             py::arg("param_dim"), py::arg("at_start"),
             "Extract a boundary face of this patch.")
        .def("eval_physical_points", [](const Patch3D2D& self, const pyck::QuadratureRule<double, 2>& quad) {
                 return self.eval_physical_points(quad);
             },
             py::arg("quadrature"),
             "Coordinate mapping of all quadrature points in active elements.")
        .def("eval_basis_functions", &Patch3D2D::eval_basis_functions,
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const Patch3D2D& p,
                 const pyck::ColMatrix<double, 2>& pts,
                 pyck::Index span,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, span, order);
             },
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_geometry", [](const Patch3D2D& self, const pyck::ColMatrix<double, 2>& pts, pyck::Index span) {
                 return self.eval_geometry(pts, span);
             },
             py::arg("params"), py::arg("span"));

    // rectangle factory
    m.def("rectangle", [](pyck::Ptr<BasisD> basis_u,
                           pyck::Ptr<BasisD> basis_v,
                           double width, double height) {
            return std::make_shared<Patch3D2D>(
                pyck::rectangle<double>(basis_u, basis_v, width, height));
        },
        py::arg("basis_u"), py::arg("basis_v"),
        py::arg("width"), py::arg("height"),
        "Create a flat rectangular surface patch in the xy-plane.");

    // === DOF Mapping ================================================================

    using DofMapper2D = pyck::DofMapper<2>;
    py::class_<DofMapper2D>(m, "DofMapper2d")
        .def("get_boundary_dofs", &DofMapper2D::get_boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("get_layer_dofs", &DofMapper2D::get_layer_dofs,
             py::arg("param_dim"), py::arg("at_start"), py::arg("layer_idx"))
        .def("get_element_dofs", static_cast<std::vector<pyck::Index> (DofMapper2D::*)(pyck::Index) const>(&DofMapper2D::get_element_dofs),
             py::arg("elem_idx"))
        .def("from_global", &DofMapper2D::from_global, py::arg("global_idx"))
        .def("num_basis", &DofMapper2D::num_basis)
        .def("degree", &DofMapper2D::degree);

    // === DOF Mapping ================================================================

    using DofMapper1D = pyck::DofMapper<1>;
    py::class_<DofMapper1D>(m, "DofMapper1d")
        .def("get_boundary_dofs", &DofMapper1D::get_boundary_dofs,
             py::arg("param_dim"), py::arg("at_start"))
        .def("get_layer_dofs", &DofMapper1D::get_layer_dofs,
             py::arg("param_dim"), py::arg("at_start"), py::arg("layer_idx"))
        .def("get_element_dofs", static_cast<std::vector<pyck::Index> (DofMapper1D::*)(pyck::Index) const>(&DofMapper1D::get_element_dofs),
             py::arg("elem_idx"))
        .def("from_global", &DofMapper1D::from_global, py::arg("global_idx"))
        .def("num_basis", &DofMapper1D::num_basis)
        .def("degree", &DofMapper1D::degree);

    // === Quadrature =================================================================

    using QR1D = pyck::QuadratureRule<double, 1>;
    py::class_<QR1D, pyck::Ptr<QR1D>>(m, "QuadratureRule1d")
        .def("points", &QR1D::points, py::return_value_policy::reference_internal)
        .def("weights", &QR1D::weights, py::return_value_policy::reference_internal)
        .def("num_points", &QR1D::num_points);

    using QR2D = pyck::QuadratureRule<double, 2>;
    py::class_<QR2D, pyck::Ptr<QR2D>>(m, "QuadratureRule2d")
        .def(py::init<const QR1D&>(), py::arg("rule"))
        .def(py::init<std::array<const QR1D*, 2>>(), py::arg("rules"))
        .def("points", &QR2D::points, py::return_value_policy::reference_internal)
        .def("weights", &QR2D::weights, py::return_value_policy::reference_internal)
        .def("num_points", &QR2D::num_points);

    using QR3D = pyck::QuadratureRule<double, 3>;
    py::class_<QR3D, pyck::Ptr<QR3D>>(m, "QuadratureRule3d")
        .def(py::init<const QR1D&>(), py::arg("rule"))
        .def(py::init<std::array<const QR1D*, 3>>(), py::arg("rules"))
        .def("points", &QR3D::points, py::return_value_policy::reference_internal)
        .def("weights", &QR3D::weights, py::return_value_policy::reference_internal)
        .def("num_points", &QR3D::num_points);

    using GL1D = pyck::GaussLegendre<double, 1>;
    py::class_<GL1D, QR1D, pyck::Ptr<GL1D>>(m, "GaussLegendre")
        .def(py::init<std::size_t>(), py::arg("num_pts"));

    using GL2D = pyck::GaussLegendre<double, 2>;
    py::class_<GL2D, QR2D, pyck::Ptr<GL2D>>(m, "GaussLegendre2d")
        .def(py::init<std::size_t>(), py::arg("num_pts"));

    using GL3D = pyck::GaussLegendre<double, 3>;
    py::class_<GL3D, QR3D, pyck::Ptr<GL3D>>(m, "GaussLegendre3d")
        .def(py::init<std::size_t>(), py::arg("num_pts"));

    // Legacy Tensor-product utilities (can be deprecated eventually)
    m.def("tensor_product_1d",
        [](const QR1D& rule) {
            std::array<const QR1D*, 1> rules = {&rule};
            return pyck::QuadratureRule<double, 1>::tensor_product(rules);
        },
        py::arg("rule"),
        "Return (points, weights) for a 1D tensor product (identity).");

    m.def("tensor_product_2d",
        [](const QR1D& rule_u, const QR1D& rule_v) {
            std::array<const QR1D*, 2> rules = {&rule_u, &rule_v};
            return pyck::QuadratureRule<double, 2>::tensor_product(rules);
        },
        py::arg("rule_u"), py::arg("rule_v"),
        "Form the 2D tensor product of two 1D quadrature rules.");

    m.def("tensor_product", &pyck::QuadratureRule<double, 2>::tensor_product,
          py::arg("rules"),
          "Form the tensor product of 2 one-dimensional quadrature rules.");

    m.def("tensor_product", &pyck::QuadratureRule<double, 3>::tensor_product,
          py::arg("rules"),
          "Form the tensor product of 3 one-dimensional quadrature rules.");

    // === Materials ==================================================================

    using Material1D = pyck::Material<double, 1>;
    py::class_<Material1D, pyck::Ptr<Material1D>>(m, "Material1d")
        .def("name", &Material1D::name)
        .def("youngs_modulus", &Material1D::youngs_modulus)
        .def("poisson_ratio", &Material1D::poisson_ratio)
        .def("shear_modulus", &Material1D::shear_modulus)
        .def("bending_stiffness", &Material1D::bending_stiffness)
        .def("shear_stiffness", &Material1D::shear_stiffness)
        .def("constitutive_matrix", &Material1D::constitutive_matrix);

    using SB1D = pyck::SlenderBeam1d<double>;
    py::class_<SB1D, Material1D, pyck::Ptr<SB1D>>(m, "SlenderBeam1d")
        .def(py::init<double, double, double, double, double>(),
             py::arg("E"), py::arg("nu"), py::arg("A"), py::arg("I"), py::arg("k") = 5.0 / 6.0)
        .def("section_area", &SB1D::section_area)
        .def("moment_inertia", &SB1D::moment_inertia)
        .def("shear_coefficient", &SB1D::shear_coefficient);

    using Material2D = pyck::Material<double, 2>;
    py::class_<Material2D, pyck::Ptr<Material2D>>(m, "Material2d")
        .def("name", &Material2D::name)
        .def("youngs_modulus", &Material2D::youngs_modulus)
        .def("poisson_ratio", &Material2D::poisson_ratio)
        .def("shear_modulus", &Material2D::shear_modulus)
        .def("bending_stiffness", &Material2D::bending_stiffness)
        .def("shear_stiffness", &Material2D::shear_stiffness)
        .def("constitutive_matrix", &Material2D::constitutive_matrix);

    using PS2D = pyck::PlaneStress2d<double>;
    py::class_<PS2D, Material2D, pyck::Ptr<PS2D>>(m, "PlaneStress2d")
        .def(py::init<double, double, double, double>(),
             py::arg("E"), py::arg("nu") = 0.3, py::arg("h") = 1.0, py::arg("k") = 5.0 / 6.0)
        .def("thickness", &PS2D::thickness)
        .def("shear_correction_factor", &PS2D::shear_correction_factor)
        .def("bending_matrix", &PS2D::bending_matrix)
        .def("shear_matrix", &PS2D::shear_matrix);

    // === Elements ===================================================================

    using Elem1D = pyck::Element<double, 1>;
    py::class_<Elem1D, pyck::Ptr<Elem1D>>(m, "Element1d")
        .def("num_node_dofs", &Elem1D::num_node_dofs)
        .def("min_order", &Elem1D::min_order)
        .def("shape_matrix", &Elem1D::shape_matrix, py::arg("shape_derivs"))
        .def("strain_displacement_matrix", &Elem1D::strain_displacement_matrix, py::arg("shape_derivs"));

    using EBB = pyck::BeamEulerBernoulli1p<double>;
    py::class_<EBB, Elem1D, pyck::Ptr<EBB>>(m, "BeamEulerBernoulli1p")
        .def(py::init<pyck::Ptr<pyck::SlenderBeam1d<double>>>(),
             py::arg("material"));

    using TBB1p = pyck::BeamTimoshenko1p<double>;
    py::class_<TBB1p, Elem1D, pyck::Ptr<TBB1p>>(m, "BeamTimoshenko1p")
        .def(py::init<pyck::Ptr<pyck::SlenderBeam1d<double>>>(),
             py::arg("material"));

    using TBB2p = pyck::BeamTimoshenko2p<double>;
    py::class_<TBB2p, Elem1D, pyck::Ptr<TBB2p>>(m, "BeamTimoshenko2p")
        .def(py::init<pyck::Ptr<pyck::SlenderBeam1d<double>>>(),
             py::arg("material"));

    // --- 2D Elements ---

    using Elem2D = pyck::Element<double, 2>;
    py::class_<Elem2D, pyck::Ptr<Elem2D>>(m, "Element2d")
        .def("num_node_dofs", &Elem2D::num_node_dofs)
        .def("min_order", &Elem2D::min_order)
        .def("shape_matrix", &Elem2D::shape_matrix, py::arg("shape_derivs"))
        .def("strain_displacement_matrix", &Elem2D::strain_displacement_matrix, py::arg("shape_derivs"));

    using KLP = pyck::PlateKirchhoffLove1p<double>;
    py::class_<KLP, Elem2D, pyck::Ptr<KLP>>(m, "PlateKirchhoffLove1p")
        .def(py::init<pyck::Ptr<pyck::PlaneStress2d<double>>>(),
             py::arg("material"));

    using RMP = pyck::PlateReissnerMindlin3p<double>;
    py::class_<RMP, Elem2D, pyck::Ptr<RMP>>(m, "PlateReissnerMindlin3p")
        .def(py::init<pyck::Ptr<pyck::PlaneStress2d<double>>>(),
             py::arg("material"));

    using RM1P = pyck::PlateReissnerMindlin1p<double>;
    py::class_<RM1P, Elem2D, pyck::Ptr<RM1P>>(m, "PlateReissnerMindlin1p")
        .def(py::init<pyck::Ptr<pyck::PlaneStress2d<double>>>(),
             py::arg("material"));

    // === Conditions =================================================================

    using CondD = pyck::Condition<double>;
    py::class_<CondD, pyck::Ptr<CondD>>(m, "Condition");

    using LC1D = pyck::LoadCondition<double, 1>;
    py::class_<LC1D, CondD, pyck::Ptr<LC1D>>(m, "LoadCondition1d")
        .def(py::init<const Patch3D1D&, const Elem1D&, const QR1D&, const pyck::Vector<double>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"), py::arg("load_values"));

    using LC2D = pyck::LoadCondition<double, 2>;
    py::class_<LC2D, CondD, pyck::Ptr<LC2D>>(m, "LoadCondition2d")
        .def(py::init<const Patch3D2D&, const Elem2D&, const QR2D&, const pyck::Vector<double>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"), py::arg("load_values"));

    // === Constraints ================================================================

    using ConstD = pyck::Constraint<double>;
    py::class_<ConstD, pyck::Ptr<ConstD>>(m, "Constraint");

    using DC = pyck::DirectConstraint<double>;
    py::class_<DC, ConstD, pyck::Ptr<DC>>(m, "DirectConstraint")
        .def(py::init<std::vector<pyck::Index>, std::vector<double>>(),
             py::arg("dofs"), py::arg("values"))
        .def(py::init<std::vector<pyck::Index>, double>(),
             py::arg("dofs"), py::arg("value") = 0.0)
        .def("dofs", &DC::dofs)
        .def("values", &DC::values)
        .def("apply", &DC::apply, py::arg("stiffness"), py::arg("load"));

    using LC = pyck::LinearConstraint<double>;
    py::class_<LC, ConstD, pyck::Ptr<LC>>(m, "LinearConstraint")
        .def(py::init<std::vector<pyck::Index>, pyck::IndexMatrix, std::vector<double>, double>(),
             py::arg("slaves"), py::arg("masters"), py::arg("weights"), py::arg("constant") = 0.0)
        .def("slaves", &LC::slaves)
        .def("masters", &LC::masters)
        .def("weights", &LC::weights)
        .def("constant", &LC::constant)
        .def("apply", &LC::apply, py::arg("stiffness"), py::arg("load"));


    // === Assembly ===================================================================

    using LEP1D = pyck::LinearElasticProblem<double, 1>;
    py::class_<LEP1D>(m, "LinearElasticProblem1d")
        .def(py::init<const pyck::Ptr<Patch3D1D>&,
                      const pyck::Ptr<Elem1D>&,
                      const pyck::Ptr<QR1D>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"))
        .def("add_condition", &LEP1D::add_condition,
             py::arg("condition"))
        .def("add_constraint", &LEP1D::add_constraint,
             py::arg("constraint"))
        .def("add_constraint", &LEP1D::add_direct_constraint,
             py::arg("direct_constraint"))
        .def("add_direct_constraint", &LEP1D::add_direct_constraint,
             py::arg("constraint"))
        .def("assemble", [](const LEP1D& p) {
            pyck::Matrix<double> K;
            pyck::Vector<double> F;
            p.assemble(K, F);
            return py::make_tuple(K, F);
        }, "Assemble global stiffness matrix and load vector. Returns (K, F).");

    using LEP2D = pyck::LinearElasticProblem<double, 2>;
    py::class_<LEP2D>(m, "LinearElasticProblem2d")
        .def(py::init<const pyck::Ptr<Patch3D2D>&,
                      const pyck::Ptr<Elem2D>&,
                      const pyck::Ptr<QR2D>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"))
        .def("add_condition", &LEP2D::add_condition,
             py::arg("condition"))
        .def("add_constraint", &LEP2D::add_constraint,
             py::arg("constraint"))
        .def("add_constraint", &LEP2D::add_direct_constraint,
             py::arg("direct_constraint"))
        .def("add_direct_constraint", &LEP2D::add_direct_constraint,
             py::arg("constraint"))
        .def("assemble", [](const LEP2D& p) {
            pyck::Matrix<double> K;
            pyck::Vector<double> F;
            p.assemble(K, F);
            return py::make_tuple(K, F);
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
        .def("knot_vector", &BSplineF::knot_vector,
             py::return_value_policy::reference_internal)
        .def("knots", [](BSplineF& bs) {
             const auto& v = bs.knots();
             return py::array_t<float>(
                 v.size(),
                 v.data(),
                 py::cast(&bs));
         });

    using Patch3D1DF = pyck::Patch<float, 1>;
    py::class_<Patch3D1DF, pyck::Ptr<Patch3D1DF>>(m, "Patch3d1d32")
        .def(py::init([](pyck::Ptr<BasisF> basis,
                         const pyck::ColMatrix<float, 3>& cp) {
                 return new Patch3D1DF(basis, cp);
             }),
             py::arg("basis"),
             py::arg("control_points"))
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
        .def("boundary", [](pyck::Ptr<Patch3D1DF> self, std::size_t param_dim, bool at_start) {
                 return pyck::create_boundary<float, 1>(self, param_dim, at_start);
             },
             py::arg("param_dim"), py::arg("at_start"),
             "Extract a boundary face of this patch.")
        .def("eval_basis_functions", &Patch3D1DF::eval_basis_functions,
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_shape_functions", [](const Patch3D1DF& p,
                 const pyck::ColMatrix<float, 1>& pts,
                 pyck::Index span,
                 std::size_t order) {
                 return p.eval_shape_functions(pts, span, order);
             },
             py::arg("params"), py::arg("span"), py::arg("order") = 0)
        .def("eval_geometry", &Patch3D1DF::eval_geometry,
             py::arg("params"), py::arg("span"));
#endif
}
