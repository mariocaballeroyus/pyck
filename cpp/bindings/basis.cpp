#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "knot_vector.hpp"
#include "bspline.hpp"
#include "nurbs.hpp"

namespace py = pybind11;

namespace pyck
{

void bind_basis(py::module_& m)
{

     // === Basis =====================================================================

     py::class_<Basis<double>, Ptr<Basis<double>>>(m, "Basis")
          // Abstract base class (no constructor)

          // Public attributes
          .def("degree", &Basis<double>::degree)
          .def("num_basis", &Basis<double>::num_basis)
          .def("num_intervals", &Basis<double>::num_intervals)
          .def("knots", [](const Basis<double>& b) {
               const auto& kv = b.knots();
               return py::array_t<double>(kv.size(), kv.data());
          })
          .def("greville_abscissae", &Basis<double>::greville_abscissae)

          // Methods
          .def("find_span", &Basis<double>::find_span,
               py::arg("u"))

          .def("eval_all",
               [](const Basis<double>& b, py::array_t<double> pts, Index order) {
                   return b.eval_all(pts.cast<Vector<double>>(), order);
               },
               py::arg("u"), py::arg("order") = 0)

          .def("insert_knot", &Basis<double>::insert_knot,
               py::arg("u"))

          .def("elevate_degree", &Basis<double>::elevate_degree);

     // === BSpline(Basis) ============================================================

     py::class_<BSpline<double>, Basis<double>, Ptr<BSpline<double>>>(m, "BSpline")
          // Factory method
          .def_static("clamped_uniform", &BSpline<double>::clamped_uniform,
               py::arg("degree"), py::arg("num_basis"));

     // === NURBS(Basis) ==============================================================

     py::class_<NURBS<double>, Basis<double>, Ptr<NURBS<double>>>(m, "NURBS")
          // Factory method
          .def_static("clamped_uniform", &NURBS<double>::clamped_uniform,
               py::arg("degree"), py::arg("num_basis"), py::arg("weights"))

          // Zero-copy view over weights with lifetime tied to NURBS
          .def("weights", &NURBS<double>::weights,
               py::return_value_policy::reference_internal);
}

}
