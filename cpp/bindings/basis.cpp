#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "knots.hpp"
#include "bspline.hpp"
#include "nurbs.hpp"

namespace py = pybind11;

namespace pyck {

// Helper function: evaluate all n basis functions at m points via eval_on_span loop.
// Returns (m × n) dense matrix by scattering span-local results.
template <std::floating_point T>
static std::vector<Matrix<T>> eval_all_aux(const Basis<T>& basis,
                                           const Vector<T>& pts,
                                           Index order)
{
    const Index n_pts = pts.size();
    const Index n = basis.num_basis();
    const Index p = basis.degree();

    std::vector<Matrix<T>> results(order + 1);
    for (Index k = 0; k <= order; ++k)
        results[k] = Matrix<T>::Zero(n_pts, n);

    for (Index i = 0; i < n_pts; ++i) {
        const Index span = basis.find_span(pts[i]);
        Vector<T> pt(1);
        pt << pts[i];
        std::vector<Matrix<T>> local;
        basis.eval_on_span(pt, span, order, local);
        for (Index k = 0; k <= order; ++k) {
            results[k].row(i).segment(span - p, p + 1) = local[k].row(0);
        }
    }

    return results;
}

void bind_basis(py::module_& m)
{

     // === Basis =====================================================================

     py::class_<Basis<double>, Ptr<Basis<double>>>(m, "Basis")
          // Abstract base class (no constructor)

          // Public attributes
          .def("degree", &Basis<double>::degree)
          .def("num_basis", &Basis<double>::num_basis)
          .def("num_intervals", &Basis<double>::num_intervals)
          .def("knots", &Basis<double>::knots,
               py::return_value_policy::reference_internal)
          .def("greville_abscissae", &Basis<double>::greville_abscissae)

          // Methods
          .def("find_span", &Basis<double>::find_span,
               py::arg("u"))

          .def("eval_on_span",
               [](const Basis<double>& b, py::array_t<double> pts, Index span, Index order) {
                   std::vector<Matrix<double>> results;
                   b.eval_on_span(pts.cast<Vector<double>>(), span, order, results);
                   return results;
               },
               py::arg("pts"), py::arg("span"), py::arg("order") = 0)

          .def("eval_all",
               [](const Basis<double>& b, py::array_t<double> pts, Index order) {
                   auto r = pts.cast<Vector<double>>();
                   return eval_all_aux(b, r, order);
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
