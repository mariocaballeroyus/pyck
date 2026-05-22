#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <array>
#include <concepts>
#include <cstddef>

#include "element.hpp"
#include "tensor_product.hpp"
#include "intrinsic_geometry.hpp"
#include "patch.hpp"
#include "eval_global_shape.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "plate_reissner_mindlin_1p.hpp"
#include "plate_reissner_mindlin_displ_2p.hpp"
#include "shell_reissner_mindlin_5p.hpp"
#include "plane_stress_2d.hpp"

namespace py = pybind11;

namespace pyck {

void bind_elements(py::module_& m)
{

    // === Element1d ==================================================================

    using Element1d = Element<double, 1>;

    py::class_<Element1d, Ptr<Element1d>>(m, "Element1d")
          .def("displacement_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       &Element1d::displacement_shape_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("rotation_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       &Element1d::rotation_shape_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("strain_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       &Element1d::strain_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("stress_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       &Element1d::stress_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"));

    // === Element2d ==================================================================

    using Element2d = Element<double, 2>;

    py::class_<Element2d, Ptr<Element2d>>(m, "Element2d")
          .def("displacement_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       &Element2d::displacement_shape_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("rotation_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       &Element2d::rotation_shape_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("strain_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       &Element2d::strain_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("stress_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       &Element2d::stress_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"));

     // === Plate elements ============================================================

     py::class_<PlateKirchhoffLove1p<double>, Element2d,
               Ptr<PlateKirchhoffLove1p<double>>>(m, "PlateKirchhoffLove1p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<PlateReissnerMindlin3p<double>, Element2d,
                Ptr<PlateReissnerMindlin3p<double>>>(m, "PlateReissnerMindlin3p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<PlateReissnerMindlinDispl2p<double>, Element2d,
                Ptr<PlateReissnerMindlinDispl2p<double>>>(m, "PlateReissnerMindlinDispl2p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<PlateReissnerMindlin1p<double>, Element2d,
                Ptr<PlateReissnerMindlin1p<double>>>(m, "PlateReissnerMindlin1p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     // === Shell elements ===========================================================

     py::class_<ShellReissnerMindlin5p<double>, Element2d,
                Ptr<ShellReissnerMindlin5p<double>>>(m, "ShellReissnerMindlin5p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));
}

}
