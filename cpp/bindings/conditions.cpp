#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "boundary_value.hpp"
#include "condition.hpp"
#include "load_boundary_condition.hpp"
#include "boundary_lagrange_condition.hpp"
#include "boundary_penalty_condition.hpp"
#include "penalty_coupling_condition.hpp"
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

    py::enum_<Field>(m, "Field")
        .value("U_X", Field::U_X)
        .value("U_Y", Field::U_Y)
        .value("U_Z", Field::U_Z)
        .value("U_N", Field::U_N)
        .value("U_S", Field::U_S)
        .value("ROT_X", Field::ROT_X)
        .value("ROT_Y", Field::ROT_Y)
        .value("ROT_Z", Field::ROT_Z)
        .value("ROT_N", Field::ROT_N)
        .value("ROT_S", Field::ROT_S);

    py::class_<BoundaryValue<double>>(m, "BoundaryValue")
        .def(py::init<Field>(), py::arg("field"));

    using LoadBoundaryCondition2d = LoadBoundaryCondition<double, 2>;
    py::class_<LoadBoundaryCondition2d, Condition<double, 2>, Ptr<LoadBoundaryCondition2d>>(m, "LoadBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LoadBoundaryCondition2d& self, BoundaryValue<double> field, double value) -> LoadBoundaryCondition2d& {
                 return self.add(std::move(field), value);
             },
             py::arg("field"), py::arg("value") = 0.0,
             py::return_value_policy::reference)
        .def("add",
             [](LoadBoundaryCondition2d& self, BoundaryValue<double> field, const Vector<double>& values) -> LoadBoundaryCondition2d& {
                 return self.add(std::move(field), values);
             },
             py::arg("field"), py::arg("values"),
             py::return_value_policy::reference);

    using PenaltyBoundaryCondition2d = PenaltyBoundaryCondition<double, 2>;
    py::class_<PenaltyBoundaryCondition2d, Condition<double, 2>, Ptr<PenaltyBoundaryCondition2d>>(m, "PenaltyBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](PenaltyBoundaryCondition2d& self, BoundaryValue<double> field,
                double penalty, double value) -> PenaltyBoundaryCondition2d& {
                 return self.add(std::move(field), penalty, value);
             },
             py::arg("field"), py::arg("penalty"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

    using PenaltyCouplingCondition2d = PenaltyCouplingCondition<double, 2>;
    py::class_<PenaltyCouplingCondition2d, Condition<double, 2>, Ptr<PenaltyCouplingCondition2d>>(m, "PenaltyCouplingCondition2d")
        .def(py::init<const PatchBoundary2d&, const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary_a"), py::arg("boundary_b"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](PenaltyCouplingCondition2d& self, BoundaryValue<double> field,
                double penalty) -> PenaltyCouplingCondition2d& {
                 return self.add(std::move(field), penalty);
             },
             py::arg("field"), py::arg("penalty"),
             py::return_value_policy::reference)
        .def("couple_displacement",
             [](PenaltyCouplingCondition2d& self, double penalty) -> PenaltyCouplingCondition2d& {
                 return self.couple_displacement(penalty);
             },
             py::arg("penalty"),
             py::return_value_policy::reference)
        .def("couple_kinematics",
             [](PenaltyCouplingCondition2d& self, double penalty_disp,
                double penalty_rot) -> PenaltyCouplingCondition2d& {
                 return self.couple_kinematics(penalty_disp, penalty_rot);
             },
             py::arg("penalty_disp"), py::arg("penalty_rot"),
             py::return_value_policy::reference);

    using LagrangeBoundaryCondition2d = LagrangeBoundaryCondition<double, 2>;
    py::class_<LagrangeBoundaryCondition2d, Condition<double, 2>, Ptr<LagrangeBoundaryCondition2d>>(m, "LagrangeBoundaryCondition2d")
        .def(py::init<const PatchBoundary2d&, const Element2d&, const QuadratureRule1d&>(),
             py::arg("boundary"), py::arg("element"), py::arg("quadrature"))
        .def("add",
             [](LagrangeBoundaryCondition2d& self, BoundaryValue<double> field, double value) -> LagrangeBoundaryCondition2d& {
                 return self.add(std::move(field), value);
             },
             py::arg("field"), py::arg("value") = 0.0,
             py::return_value_policy::reference);

}

}
