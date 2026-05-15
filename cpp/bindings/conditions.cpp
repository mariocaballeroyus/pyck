#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "load_condition.hpp"
#include "load_boundary_condition.hpp"
#include "lagrange_boundary_condition.hpp"
#include "lagrange_domain_condition.hpp"
#include "nitsche_boundary_condition.hpp"
#include "penalty_boundary_condition.hpp"
#include "penalty_coupling_condition.hpp"
#include "lagrange_coupling_condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"

namespace py = pybind11;

namespace pyck {

void bind_conditions(py::module_& m)
{
    using Patch1d = Patch<double, 1>;
    using Patch2d = Patch<double, 2>;
    using PatchBoundary2d = PatchBoundary<double, 2>;
    using Element1d = Element<double, 1>;
    using Element2d = Element<double, 2>;
    using QuadratureRule1d = QuadratureRule<double, 1>;
    using QuadratureRule2d = QuadratureRule<double, 2>;

    py::class_<Condition<double>, Ptr<Condition<double>>>(m, "Condition");

    py::class_<BoundaryField<double>, Ptr<BoundaryField<double>>>(m, "BoundaryField");

    py::class_<TransverseDisplacement<double>, BoundaryField<double>,
               Ptr<TransverseDisplacement<double>>>(m, "TransverseDisplacement")
        .def(py::init<>());

    py::class_<NormalRotation<double>, BoundaryField<double>,
               Ptr<NormalRotation<double>>>(m, "NormalRotation")
        .def(py::init<>());

    py::class_<TangentialRotation<double>, BoundaryField<double>,
               Ptr<TangentialRotation<double>>>(m, "TangentialRotation")
        .def(py::init<>());

    py::class_<NormalTransverseShear<double>, BoundaryField<double>,
               Ptr<NormalTransverseShear<double>>>(m, "NormalTransverseShear")
        .def(py::init<>());

    py::class_<NormalBendingMoment<double>, BoundaryField<double>,
               Ptr<NormalBendingMoment<double>>>(m, "NormalBendingMoment")
        .def(py::init<>());

    py::class_<TwistingMoment<double>, BoundaryField<double>,
               Ptr<TwistingMoment<double>>>(m, "TwistingMoment")
        .def(py::init<>());

    py::class_<BasisValue<double>, BoundaryField<double>,
               Ptr<BasisValue<double>>>(m, "BasisValue")
        .def(py::init<std::size_t>(), py::arg("dof_index") = 0);

    py::class_<BasisNormalSlope<double>, BoundaryField<double>,
               Ptr<BasisNormalSlope<double>>>(m, "BasisNormalSlope")
        .def(py::init<std::size_t>(), py::arg("dof_index") = 0);

    py::class_<BasisNormalCurvature<double>, BoundaryField<double>,
               Ptr<BasisNormalCurvature<double>>>(m, "BasisNormalCurvature")
        .def(py::init<std::size_t>(), py::arg("dof_index") = 0);

    using LoadCondition1d = LoadCondition<double, 1>;
    py::class_<LoadCondition1d, Condition<double>, Ptr<LoadCondition1d>>(m, "LoadCondition1d")
        .def(py::init<const Patch1d&, const Element1d&, const QuadratureRule1d&, const Vector<double>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"), py::arg("load_values"));

    using LoadCondition2d = LoadCondition<double, 2>;
    py::class_<LoadCondition2d, Condition<double>, Ptr<LoadCondition2d>>(m, "LoadCondition2d")
        .def(py::init<const Patch2d&, const Element2d&, const QuadratureRule2d&, const Vector<double>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"), py::arg("load_values"));

    using LoadBoundaryCondition2d = LoadBoundaryCondition<double, 2>;
    py::class_<LoadBoundaryCondition2d, Condition<double>, Ptr<LoadBoundaryCondition2d>>(m, "LoadBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LoadBoundaryCondition2d& self, Ptr<const BoundaryField<double>> field, double value) -> LoadBoundaryCondition2d& {
                 return self.add(std::move(field), value);
             },
             py::arg("field"), py::arg("value") = 0.0,
             py::return_value_policy::reference)
        .def("add",
             [](LoadBoundaryCondition2d& self, Ptr<const BoundaryField<double>> field, const Vector<double>& values) -> LoadBoundaryCondition2d& {
                 return self.add(std::move(field), values);
             },
             py::arg("field"), py::arg("values"),
             py::return_value_policy::reference);

    using PenaltyBoundaryCondition2d = PenaltyBoundaryCondition<double, 2>;
    py::class_<PenaltyBoundaryCondition2d, Condition<double>, Ptr<PenaltyBoundaryCondition2d>>(m, "PenaltyBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](PenaltyBoundaryCondition2d& self, Ptr<const BoundaryField<double>> field,
                double penalty, double value) -> PenaltyBoundaryCondition2d& {
                 return self.add(std::move(field), penalty, value);
             },
             py::arg("field"), py::arg("penalty"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using LagrangeBoundaryCondition2d = LagrangeBoundaryCondition<double, 2>;
    py::class_<LagrangeBoundaryCondition2d, Condition<double>, Ptr<LagrangeBoundaryCondition2d>>(m, "LagrangeBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LagrangeBoundaryCondition2d& self, Ptr<const BoundaryField<double>> field, double value) -> LagrangeBoundaryCondition2d& {
                 return self.add(std::move(field), value);
             },
             py::arg("field"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using LagrangeDomainCondition2d = LagrangeDomainCondition<double, 2>;
    py::class_<LagrangeDomainCondition2d, Condition<double>, Ptr<LagrangeDomainCondition2d>>(m, "LagrangeDomainCondition2d")
        .def(py::init<const Patch2d&, const Element2d&, const QuadratureRule2d&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LagrangeDomainCondition2d& self, std::size_t dof_index, double value) -> LagrangeDomainCondition2d& {
                 return self.add(dof_index, value);
             },
             py::arg("dof_index"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using NitscheBoundaryCondition2d = NitscheBoundaryCondition<double, 2>;
    py::class_<NitscheBoundaryCondition2d, Condition<double>, Ptr<NitscheBoundaryCondition2d>>(m, "NitscheBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&, double>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"),
             py::arg("thickness"))
        .def("add",
             [](NitscheBoundaryCondition2d& self,
                Ptr<const BoundaryField<double>> displacement_field,
                Ptr<const BoundaryField<double>> traction_field,
                double penalty, double value) -> NitscheBoundaryCondition2d& {
                 return self.add(std::move(displacement_field),
                                 std::move(traction_field), penalty, value);
             },
             py::arg("displacement_field"), py::arg("traction_field"),
             py::arg("penalty"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using PenaltyCouplingCondition2d = PenaltyCouplingCondition<double, 2>;
    py::class_<PenaltyCouplingCondition2d, Condition<double>, Ptr<PenaltyCouplingCondition2d>>(m, "PenaltyCouplingCondition2d")
        .def(py::init<const PatchBoundary2d&,
                      const PatchBoundary2d&,
                      std::size_t,
                      std::size_t,
                      const Element2d&,
                      const Element2d&,
                      const QuadratureRule1d&,
                      bool>(),
             py::arg("side_a"), py::arg("side_b"),
             py::arg("patch_a_idx"), py::arg("patch_b_idx"),
             py::arg("element_a"), py::arg("element_b"),
             py::arg("quadrature"), py::arg("reverse") = false)
        .def("add",
             [](PenaltyCouplingCondition2d& self,
                Ptr<const BoundaryField<double>> field_a,
                Ptr<const BoundaryField<double>> field_b,
                double penalty, double sign_b, double value) -> PenaltyCouplingCondition2d& {
                 return self.add(std::move(field_a), std::move(field_b),
                                 penalty, sign_b, value);
             },
             py::arg("field_a"), py::arg("field_b"),
             py::arg("penalty"), py::arg("sign_b") = 1.0, py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using LagrangeCouplingCondition2d = LagrangeCouplingCondition<double, 2>;
    py::class_<LagrangeCouplingCondition2d, Condition<double>, Ptr<LagrangeCouplingCondition2d>>(m, "LagrangeCouplingCondition2d")
        .def(py::init<const PatchBoundary2d&,
                      const PatchBoundary2d&,
                      std::size_t,
                      std::size_t,
                      const Element2d&,
                      const Element2d&,
                      const QuadratureRule1d&,
                      bool>(),
             py::arg("side_a"), py::arg("side_b"),
             py::arg("patch_a_idx"), py::arg("patch_b_idx"),
             py::arg("element_a"), py::arg("element_b"),
             py::arg("quadrature"), py::arg("reverse") = false)
        .def("add",
             [](LagrangeCouplingCondition2d& self,
                Ptr<const BoundaryField<double>> field_a,
                Ptr<const BoundaryField<double>> field_b,
                double sign_b, double value) -> LagrangeCouplingCondition2d& {
                 return self.add(std::move(field_a), std::move(field_b),
                                 sign_b, value);
             },
             py::arg("field_a"), py::arg("field_b"),
             py::arg("sign_b") = 1.0, py::arg("value") = 0.0,
             py::return_value_policy::reference);
}

}
