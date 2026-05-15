#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "knots.hpp"
#include "bspline.hpp"
#include "nurbs.hpp"

namespace py = pybind11;

namespace pyck {

void bind_basis(py::module_& m)
{

     // === KnotVector ================================================================

     py::class_<KnotVector<double>>(m, "KnotVector")
          // Constructor
          .def(py::init<Vector<double>>(),
               py::arg("knots"))

          // Zero-copy view over knots with lifetime tied to KnotVector
          .def("knots", &KnotVector<double>::data,
               py::return_value_policy::reference_internal)

          .def("insert", &KnotVector<double>::insert,
               py::arg("u"), py::arg("count") = 1)

          .def("elevate", &KnotVector<double>::elevate,
               py::arg("count") = 1);

     // === Basis =====================================================================

     py::class_<KnotInsertion<double>>(m, "KnotInsertion")
          .def_readonly("basis",     &KnotInsertion<double>::basis)
          .def_readonly("transform", &KnotInsertion<double>::transform);

     py::class_<DegreeElevation<double>>(m, "DegreeElevation")
          .def_readonly("basis",     &DegreeElevation<double>::basis)
          .def_readonly("transform", &DegreeElevation<double>::transform);

     py::class_<Basis<double>, Ptr<Basis<double>>>(m, "Basis")
          // Abstract base class (no constructor)

          // Public attributes
          .def("degree", &Basis<double>::degree)
          .def("num_basis", &Basis<double>::num_basis)
          .def("greville_abscissae", &Basis<double>::greville_abscissae)

          // Methods
          .def("eval_all", &Basis<double>::eval_all,
               py::arg("u"),
               py::arg("order"))

          .def("insert_knot", &Basis<double>::insert_knot,
               py::arg("u"), py::arg("count") = 1)

          .def("elevate_degree", &Basis<double>::elevate_degree,
               py::arg("count") = 1);

     // === BSpline(Basis) ============================================================

     py::class_<BSpline<double>, Basis<double>, Ptr<BSpline<double>>>(m, "BSpline")
          // Constructor
          .def(py::init<std::size_t, KnotVector<double>>(),
               py::arg("degree"), 
               py::arg("knot_vector"));

     // === NURBS(Basis) ==============================================================

     py::class_<NURBS<double>, Basis<double>, Ptr<NURBS<double>>>(m, "NURBS")
          // Constructor
          .def(py::init<std::size_t, KnotVector<double>, Vector<double>>(),
               py::arg("degree"),
               py::arg("knot_vector"),
               py::arg("weights"))
          
          // Zero-copy view over weights with lifetime tied to NURBS
          .def("weights", &NURBS<double>::weights,
               py::return_value_policy::reference_internal);
}

}
