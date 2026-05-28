#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "boundary_field.hpp"
#include "condition.hpp"
#include "load_boundary_condition.hpp"
#include "boundary_lagrange_condition.hpp"
#include "boundary_penalty_condition.hpp"
#include "patch_boundary.hpp"
#include "element.hpp"
#include "quadrature.hpp"

namespace py = pybind11;

namespace pyck {

void bind_conditions(py::module_& m)
{
    using PatchBoundary2d = PatchBoundary<double, 2>;
    using Element2d = Element<double, 2>;
    using QuadratureRule1d = QuadratureRule<double, 1>;

    py::class_<Condition<double, 1>, Ptr<Condition<double, 1>>>(m, "Condition1d");
    py::class_<Condition<double, 2>, Ptr<Condition<double, 2>>>(m, "Condition2d");

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

    using LoadBoundaryCondition2d = LoadBoundaryCondition<double, 2>;
    py::class_<LoadBoundaryCondition2d, Condition<double, 2>, Ptr<LoadBoundaryCondition2d>>(m, "LoadBoundaryCondition2d")
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
    py::class_<PenaltyBoundaryCondition2d, Condition<double, 2>, Ptr<PenaltyBoundaryCondition2d>>(m, "PenaltyBoundaryCondition2d")
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
    py::class_<LagrangeBoundaryCondition2d, Condition<double, 2>, Ptr<LagrangeBoundaryCondition2d>>(m, "LagrangeBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LagrangeBoundaryCondition2d& self, Ptr<const BoundaryField<double>> field, double value) -> LagrangeBoundaryCondition2d& {
                 return self.add(std::move(field), value);
             },
             py::arg("field"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

}

}
