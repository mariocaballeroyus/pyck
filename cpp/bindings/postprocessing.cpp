#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "function.hpp"
#include "field_transform.hpp"
#include "inner_product.hpp"

namespace py = pybind11;

namespace pyck {

void bind_postprocessing(py::module_& m)
{
    // === FieldType ================================================================
    py::enum_<FieldType>(m, "FieldType")
        .value("PRIMAL",       FieldType::PRIMAL)
        .value("DISPLACEMENT", FieldType::DISPLACEMENT)
        .value("ROTATION",     FieldType::ROTATION)
        .value("STRAIN",       FieldType::STRAIN)
        .value("TRACTION",     FieldType::TRACTION)
        .value("MOMENT",       FieldType::MOMENT);

    // === Function =================================================================
    py::class_<Function<double, 1>, Ptr<Function<double, 1>>>(m, "Function1d")
        .def(py::init<Vector<double>,
                      Ptr<Element<double, 1>>,
                      Ptr<Patch<double, 1>>,
                      FieldType>(),
             py::arg("u"),
             py::arg("element"),
             py::arg("patch"),
             py::arg("field"))
        .def("__call__", &Function<double, 1>::operator(), py::arg("params"));

    py::class_<Function<double, 2>, Ptr<Function<double, 2>>>(m, "Function2d")
        .def(py::init<Vector<double>,
                      Ptr<Element<double, 2>>,
                      Ptr<Patch<double, 2>>,
                      FieldType>(),
             py::arg("u"),
             py::arg("element"),
             py::arg("patch"),
             py::arg("field"))
        .def("__call__", &Function<double, 2>::operator(), py::arg("params"));

    // === Quadrature data + inner product ==========================================
    m.def("eval_quadrature_data",
          &eval_quadrature_data<double, 1>,
          py::arg("patch"), py::arg("quadrature"));
    m.def("eval_quadrature_data",
          &eval_quadrature_data<double, 2>,
          py::arg("patch"), py::arg("quadrature"));

    m.def("inner_product",
          &inner_product<double, 1>,
          py::arg("patch"), py::arg("quadrature"),
          py::arg("f_vals"), py::arg("g_vals"));
    m.def("inner_product",
          &inner_product<double, 2>,
          py::arg("patch"), py::arg("quadrature"),
          py::arg("f_vals"), py::arg("g_vals"));

    // === Field transforms =========================================================
    m.def("covariant_to_cartesian",
          &covariant_to_cartesian<double>,
          py::arg("patch"), py::arg("params"), py::arg("comps"));
}

}
