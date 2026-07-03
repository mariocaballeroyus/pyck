#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <array>
#include <concepts>
#include <cstddef>

#include "element.hpp"
#include "tensor_product.hpp"
#include "primitives_intrinsic.hpp"
#include "patch.hpp"
#include "eval_global_shape.hpp"
#include "shell_reissner_mindlin_5p.hpp"
#include "shell_reissner_mindlin_4p.hpp"
#include "shell_reissner_mindlin_hier_4p.hpp"
#include "shell_reissner_mindlin_hier_5p.hpp"
#include "shell_reissner_mindlin_hier_5p_helmholtz.hpp"
#include "shell_reissner_mindlin_hier_disp_5p.hpp"
#include "shell_kirchhoff_love_3p.hpp"
#include "mixed_membrane_strain_shell.hpp"
#include "plane_stress_2d.hpp"
#include "quadrature.hpp"

namespace py = pybind11;

namespace pyck {

void bind_elements(py::module_& m)
{

    // === Element1d ==================================================================

    using Element1d = Element<double, 1>;

    py::class_<Element1d, Ptr<Element1d>>(m, "Element1d")
          .def("primal_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       [](const Element1d& e, const ElementValues<double, 1>& ev) -> const Matrix<double>& {
                           e.primal_shape_matrix(ev);
                           return e.N_primal_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("displacement_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       [](const Element1d& e, const ElementValues<double, 1>& ev) -> const Matrix<double>& {
                           e.displacement_shape_matrix(ev);
                           return e.N_w_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("rotation_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       [](const Element1d& e, const ElementValues<double, 1>& ev) -> const Matrix<double>& {
                           e.rotation_shape_matrix(ev);
                           return e.N_phi_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("strain_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       [](const Element1d& e, const ElementValues<double, 1>& ev) -> const Matrix<double>& {
                           e.strain_matrix(ev);
                           return e.B_voigt_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("stress_shape_matrix",
               [](const Element1d& elem,
                  const Patch<double, 1>& patch,
                  const ColMatrix<double, 1>& params) -> Matrix<double> {
                   return eval_global_shape<double, 1>(patch, elem,
                       [](const Element1d& e, const ElementValues<double, 1>& ev) -> const Matrix<double>& {
                           e.stress_shape_matrix(ev);
                           return e.N_sigma_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"));

    // === Element2d ==================================================================

    using Element2d = Element<double, 2>;

    py::class_<Element2d, Ptr<Element2d>>(m, "Element2d")
          .def("primal_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       [](const Element2d& e, const ElementValues<double, 2>& ev) -> const Matrix<double>& {
                           e.primal_shape_matrix(ev);
                           return e.N_primal_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("displacement_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       [](const Element2d& e, const ElementValues<double, 2>& ev) -> const Matrix<double>& {
                           e.displacement_shape_matrix(ev);
                           return e.N_w_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("rotation_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       [](const Element2d& e, const ElementValues<double, 2>& ev) -> const Matrix<double>& {
                           e.rotation_shape_matrix(ev);
                           return e.N_phi_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("strain_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       [](const Element2d& e, const ElementValues<double, 2>& ev) -> const Matrix<double>& {
                           e.strain_matrix(ev);
                           return e.B_voigt_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"))

          .def("stress_shape_matrix",
               [](const Element2d& elem,
                  const Patch<double, 2>& patch,
                  const ColMatrix<double, 2>& params) -> Matrix<double> {
                   return eval_global_shape<double, 2>(patch, elem,
                       [](const Element2d& e, const ElementValues<double, 2>& ev) -> const Matrix<double>& {
                           e.stress_shape_matrix(ev);
                           return e.N_sigma_;
                       }, params);
               },
               py::arg("patch"),
               py::arg("params"));

     // === Shell elements ===========================================================

     py::class_<ShellReissnerMindlin5p<double>, Element2d,
                Ptr<ShellReissnerMindlin5p<double>>>(m, "ShellReissnerMindlin5p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellReissnerMindlin4p<double>, Element2d,
                Ptr<ShellReissnerMindlin4p<double>>>(m, "ShellReissnerMindlin4p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellReissnerMindlinHier4p<double>, Element2d,
                Ptr<ShellReissnerMindlinHier4p<double>>>(m, "ShellReissnerMindlinHier4p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellReissnerMindlinHier5p<double>, Element2d,
                Ptr<ShellReissnerMindlinHier5p<double>>>(m, "ShellReissnerMindlinHier5p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellReissnerMindlinHier5pHelmholtz<double>, Element2d,
                Ptr<ShellReissnerMindlinHier5pHelmholtz<double>>>(m, "ShellReissnerMindlinHier5pHelmholtz")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellReissnerMindlinHierDisp5p<double>, Element2d,
                Ptr<ShellReissnerMindlinHierDisp5p<double>>>(m, "ShellReissnerMindlinHierDisp5p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<ShellKirchhoffLove3p<double>, Element2d,
                Ptr<ShellKirchhoffLove3p<double>>>(m, "ShellKirchhoffLove3p")
          .def(py::init<Ptr<PlaneStress2d<double>>>(),
               py::arg("material"));

     py::class_<MixedMembraneStrainShell<double>, Element2d,
                Ptr<MixedMembraneStrainShell<double>>>(m, "MixedMembraneStrainShell")
          .def(py::init<Ptr<Patch<double, 2>>, Ptr<Element<double, 2>>,
                        Ptr<QuadratureRule<double, 2>>, Index>(),
               py::arg("patch"), py::arg("base_element"), py::arg("quadrature"),
               py::arg("degree_drop") = 1)
          .def("recover_membrane_force",
               &MixedMembraneStrainShell<double>::recover_membrane_force,
               py::arg("full_u"), py::arg("params"));
}

}
