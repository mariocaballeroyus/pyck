#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "bspline.hpp"
#include "patch_boundary.hpp"
#include "basis_values.hpp"
#include "intrinsic_geometry.hpp"

namespace py = pybind11;

namespace pyck {

void bind_geometry(py::module_& m)
{

    // === Patch ======================================================================

    using Patch1d = Patch<double, 1>;

    py::class_<Patch1d, Ptr<Patch1d>>(m, "Patch1d")
        // Constructor
        .def(py::init<Ptr<Basis<double>>, const ColMatrix<double, 3>&>(),
             py::arg("basis"),
             py::arg("control_points"))

        // Zero-copy view over control points with lifetime tied to Patch
        .def("control_pts",
            static_cast<const ColMatrix<double, 3>& (Patch1d::*)() const>(&Patch1d::control_pts),
            py::return_value_policy::reference_internal)

        .def("eval_physical",
            static_cast<ColMatrix<double, 3> (Patch1d::*)(const ColMatrix<double, 1>&) const>(&Patch1d::eval_physical),
            py::arg("pts"))

        .def("insert_knot",
            static_cast<Patch1d (Patch1d::*)(double) const>(&Patch1d::insert_knot),
            py::arg("u"))

        .def("elevate_degree",
            static_cast<Patch1d (Patch1d::*)() const>(&Patch1d::elevate_degree));

    using Patch2d = Patch<double, 2>;

    py::class_<Patch2d, Ptr<Patch2d>>(m, "Patch2d")
        // Constructor
        .def(py::init<Ptr<Basis<double>>, Ptr<Basis<double>>, const ColMatrix<double, 3>&>(),
             py::arg("basis_u"),
             py::arg("basis_v"),
             py::arg("control_points"))

        // Public attributes
        .def("control_pts",
            static_cast<const ColMatrix<double, 3>& (Patch2d::*)() const>(&Patch2d::control_pts),
            py::return_value_policy::reference_internal)

        .def("eval_physical",
            static_cast<ColMatrix<double, 3> (Patch2d::*)(const ColMatrix<double, 2>&) const>(&Patch2d::eval_physical),
            py::arg("pts"))

        .def("insert_knot",
            static_cast<Patch2d (Patch2d::*)(std::size_t, double) const>(&Patch2d::insert_knot),
            py::arg("dir"), py::arg("u"))

        .def("elevate_degree",
            static_cast<Patch2d (Patch2d::*)(std::size_t) const>(&Patch2d::elevate_degree),
            py::arg("dir"));

    // === Patch Boundary =============================================================

    using PatchBoundary2d = PatchBoundary<double, 2>;

    py::class_<PatchBoundary2d, Patch1d, Ptr<PatchBoundary2d>>(m, "PatchBoundary2d")
        // Constructor
        .def(py::init<const Ptr<Patch2d>&, std::size_t, bool>(),
            py::arg("parent"),
            py::arg("param_dim"),
            py::arg("at_start"))

        // Public attributes
        .def("param_dim", &PatchBoundary2d::param_dim)
        .def("at_start", &PatchBoundary2d::at_start)

        // DOF accessors
        .def("displacement_dofs", &PatchBoundary2d::displacement_dofs)
        .def("rotation_dofs", &PatchBoundary2d::rotation_dofs);

}

}
