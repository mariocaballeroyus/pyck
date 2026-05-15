#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "direct_constraint.hpp"
#include "linear_constraint.hpp"

namespace py = pybind11;

namespace pyck {

void bind_constraints(py::module_& m)
{
    py::class_<Constraint<double>, Ptr<Constraint<double>>>(m, "Constraint");

    py::class_<DirectConstraint<double>, Constraint<double>,
              Ptr<DirectConstraint<double>>>(m, "DirectConstraint")

         // Constructors
         .def(py::init<IndexVector, Vector<double>>(),
              py::arg("dofs"),
              py::arg("values"))
         .def(py::init<IndexVector, double>(),
              py::arg("dofs"),
              py::arg("value"))

         // Zero-copy views over member arrays, lifetime tied to constraint
         .def("dofs", &DirectConstraint<double>::dofs,
              py::return_value_policy::reference_internal)
         .def("values", &DirectConstraint<double>::values,
              py::return_value_policy::reference_internal);

    py::class_<LinearConstraint<double>, Constraint<double>,
              Ptr<LinearConstraint<double>>>(m, "LinearConstraint")

         // Constructor
         .def(py::init<IndexVector, IndexMatrix, Vector<double>, double>(),
              py::arg("slaves"),
              py::arg("masters"),
              py::arg("weights"),
              py::arg("constant"))

         // Zero-copy views over member arrays, lifetime tied to constraint
         .def("slaves", &LinearConstraint<double>::slaves,
              py::return_value_policy::reference_internal)
         .def("masters", &LinearConstraint<double>::masters,
              py::return_value_policy::reference_internal)
         .def("weights", &LinearConstraint<double>::weights,
              py::return_value_policy::reference_internal)
         .def("constant", &LinearConstraint<double>::constant);
}

}
