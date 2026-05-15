#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "physical_points.hpp"
#include "compute_l2_error.hpp"

namespace py = pybind11;

namespace pyck {

void bind_postprocessing(py::module_& m)
{

      // === Evaluate Physical Points =================================================
      m.def("eval_physical_points",
          &eval_physical_points<double, 1>,
          py::arg("patch"), 
          py::arg("quadrature"));

      m.def("eval_physical_points",
          &eval_physical_points<double, 2>,
          py::arg("patch"),
          py::arg("quadrature"));

      // === Evaluate Parametric Points ===============================================
      m.def("eval_parametric_points",
          &eval_parametric_points<double, 1>,
          py::arg("patch"),
          py::arg("quadrature"));

      m.def("eval_parametric_points",
          &eval_parametric_points<double, 2>,
          py::arg("patch"),
          py::arg("quadrature"));

      // === Evaluate Integration Measures ============================================
      m.def("eval_integration_measures",
          &eval_integration_measures<double, 1>,
          py::arg("patch"),
          py::arg("quadrature"));

      m.def("eval_integration_measures",
          &eval_integration_measures<double, 2>,
          py::arg("patch"),
          py::arg("quadrature"));

      // === Compute L2 Error =========================================================

      m.def("compute_l2_error",
          &compute_l2_error<double, 1>,
          py::arg("patch"),
          py::arg("element"),
          py::arg("quadrature"),
          py::arg("order"),
          py::arg("num_u"),
          py::arg("integrand"));
          
      m.def("compute_l2_error",
          &compute_l2_error<double, 2>,
          py::arg("patch"),
          py::arg("element"),
          py::arg("quadrature"),
          py::arg("order"),
          py::arg("num_u"),
          py::arg("integrand"));
}

}
