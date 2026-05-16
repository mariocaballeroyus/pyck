#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <array>
#include <concepts>
#include <cstddef>

#include "element.hpp"
#include "basis_derivs.hpp"
#include "local_frame.hpp"
#include "patch.hpp"
#include "christoffels.hpp"
#include "beam_euler_bernoulli_1p.hpp"
#include "beam_timoshenko_1p.hpp"
#include "beam_timoshenko_2p.hpp"
#include "plate_kirchhoff_love_1p.hpp"
#include "plate_reissner_mindlin_3p.hpp"
#include "plate_reissner_mindlin_1p.hpp"
#include "plate_reissner_mindlin_displ_2p.hpp"
#include "plate_reissner_mindlin_displ_3p.hpp"
#include "shell_reissner_mindlin_5p.hpp"
#include "uniaxial_stress_1d.hpp"
#include "plane_stress_2d.hpp"

namespace py = pybind11;

namespace pyck {

namespace {

/// @brief Pointer-to-member-function type for an element shape-matrix method.
template <std::floating_point T, std::size_t d>
using ShapeMatrixFn = Matrix<T> (Element<T, d>::*)(const Patch<T, d>&,
                                                   const BasisDerivs<T, d>&,
                                                   const LocalFrame<T, d>&,
                                                   const ChristoffelSymbols<T, d>&) const;

/**
 * @brief Evaluate a global shape matrix at a set of global parametric points.
 *
 * Used exclusively by the Python bindings to drive the per-point loop in C++.
 */
template <std::floating_point T, std::size_t d>
Matrix<T> eval_global_shape(
    const Patch<T, d>& patch,
    const Element<T, d>& element,
    ShapeMatrixFn<T, d> mem_fn,
    const ColMatrix<T, d>& params)
{
    const Index n_pts    = static_cast<Index>(params.rows());
    const Index ndof     = static_cast<Index>(element.num_node_dofs());
    const Index n_global = static_cast<Index>(patch.num_control_pts()) * ndof;
    const std::size_t order = element.min_order();

    // Number of knot spans per direction — matches the U-inner convention
    // used by Patch::decode_span and DofMapper::get_element_dofs.
    std::array<Index, d> n_spans;
    for (std::size_t dir = 0; dir < d; ++dir)
        n_spans[dir] = static_cast<Index>(
            patch.basis(dir).knot_vector().num_spans());

    Matrix<T> N_global;
    Index n_rows_per_pt = 0;
    bool initialized = false;

    for (Index i = 0; i < n_pts; ++i)
    {
        ColMatrix<T, d> pt = params.row(i);

        std::array<Index, d> span_idx;
        for (std::size_t dir = 0; dir < d; ++dir)
            span_idx[dir] = patch.basis(dir).find_span(
                static_cast<T>(pt(0, static_cast<Index>(dir))));

        Index flat_span = span_idx[0];
        if constexpr (d >= 2)
            flat_span += span_idx[1] * n_spans[0];

        auto bd      = eval_basis(patch, pt, flat_span, order);
        auto act_pts = patch.active_control_pts(flat_span);
        auto lf      = eval_local_frame(bd, act_pts);
        auto chr     = eval_christoffel(lf);

        const Matrix<T> N_span = (element.*mem_fn)(patch, bd, lf, chr);

        if (!initialized)
        {
            n_rows_per_pt = static_cast<Index>(N_span.rows());
            N_global = Matrix<T>::Zero(n_pts * n_rows_per_pt, n_global);
            initialized = true;
        }

        const auto active_cp = patch.dof_mapper().get_element_dofs(flat_span);
        const Index n_active = static_cast<Index>(active_cp.size());

        const Index row_start = i * n_rows_per_pt;
        for (Index k = 0; k < n_active; ++k)
        {
            for (Index v = 0; v < ndof; ++v)
            {
                N_global.block(row_start, active_cp[k] * ndof + v, n_rows_per_pt, 1)
                    .noalias() = N_span.col(k * ndof + v);
            }
        }
    }

    if (!initialized)
        return Matrix<T>(0, n_global);
    return N_global;
}

/// @brief Pointer-to-member-function type for element matrix methods without Christoffel symbols.
template <std::floating_point T, std::size_t d>
using SimpleMatrixFn = Matrix<T> (Element<T, d>::*)(
    const Patch<T, d>&,
    const BasisDerivs<T, d>&,
    const LocalFrame<T, d>&) const;

/**
 * @brief Evaluate a global matrix (stress_matrix) at a set of global parametric points.
 *
 * Used exclusively by the Python bindings for methods that don't need Christoffel symbols.
 */
template <std::floating_point T, std::size_t d>
Matrix<T> eval_global_simple(
    const Patch<T, d>& patch,
    const Element<T, d>& element,
    SimpleMatrixFn<T, d> mem_fn,
    const ColMatrix<T, d>& params)
{
    const Index n_pts    = static_cast<Index>(params.rows());
    const Index ndof     = static_cast<Index>(element.num_node_dofs());
    const Index n_global = static_cast<Index>(patch.num_control_pts()) * ndof;
    const std::size_t order = element.min_order();

    // Number of knot spans per direction — matches the U-inner convention
    // used by Patch::decode_span and DofMapper::get_element_dofs.
    std::array<Index, d> n_spans;
    for (std::size_t dir = 0; dir < d; ++dir)
        n_spans[dir] = static_cast<Index>(
            patch.basis(dir).knot_vector().num_spans());

    Matrix<T> N_global;
    Index n_rows_per_pt = 0;
    bool initialized = false;

    for (Index i = 0; i < n_pts; ++i)
    {
        ColMatrix<T, d> pt = params.row(i);

        std::array<Index, d> span_idx;
        for (std::size_t dir = 0; dir < d; ++dir)
            span_idx[dir] = patch.basis(dir).find_span(
                static_cast<T>(pt(0, static_cast<Index>(dir))));

        Index flat_span = span_idx[0];
        if constexpr (d >= 2)
            flat_span += span_idx[1] * n_spans[0];

        auto bd      = eval_basis(patch, pt, flat_span, order);
        auto act_pts = patch.active_control_pts(flat_span);
        auto lf      = eval_local_frame(bd, act_pts);

        const Matrix<T> N_span = (element.*mem_fn)(patch, bd, lf);

        if (!initialized)
        {
            n_rows_per_pt = static_cast<Index>(N_span.rows());
            N_global = Matrix<T>::Zero(n_pts * n_rows_per_pt, n_global);
            initialized = true;
        }

        const auto active_cp = patch.dof_mapper().get_element_dofs(flat_span);
        const Index n_active = static_cast<Index>(active_cp.size());

        const Index row_start = i * n_rows_per_pt;
        for (Index k = 0; k < n_active; ++k)
        {
            for (Index v = 0; v < ndof; ++v)
            {
                N_global.block(row_start, active_cp[k] * ndof + v, n_rows_per_pt, 1)
                    .noalias() = N_span.col(k * ndof + v);
            }
        }
    }

    if (!initialized)
        return Matrix<T>(0, n_global);
    return N_global;
}

} // namespace

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
                   return eval_global_simple<double, 1>(patch, elem,
                       &Element1d::stress_matrix, params);
               },
               py::arg("patch"),
               py::arg("params"));

     // === Beam elements ===========================================================

     py::class_<BeamEulerBernoulli1p<double>, Element1d,
                Ptr<BeamEulerBernoulli1p<double>>>(m, "BeamEulerBernoulli1p")
          .def(py::init<Ptr<UniaxialStress1d<double>>>(),
               py::arg("material"));

     py::class_<BeamTimoshenko1p<double>, Element1d,
                Ptr<BeamTimoshenko1p<double>>>(m, "BeamTimoshenko1p")
          .def(py::init<Ptr<UniaxialStress1d<double>>>(),
               py::arg("material"));

     py::class_<BeamTimoshenko2p<double>, Element1d,
                Ptr<BeamTimoshenko2p<double>>>(m, "BeamTimoshenko2p")
          .def(py::init<Ptr<UniaxialStress1d<double>>>(),
               py::arg("material"));

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
                   return eval_global_simple<double, 2>(patch, elem,
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

     py::class_<PlateReissnerMindlinDispl3p<double>, Element2d,
                Ptr<PlateReissnerMindlinDispl3p<double>>>(m, "PlateReissnerMindlinDispl3p")
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
