#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <map>
#include <string>
#include <vector>

#include "vtk_export.hpp"
#include "vtk_multiblock.hpp"

namespace py = pybind11;

namespace pyck
{

void bind_io(py::module_& m)
{
    m.def("export_bezier_vtu", &io::export_bezier_vtu<double>,
          py::arg("path"),
          py::arg("patches"),
          py::arg("point_data"),
          py::arg("point_data_extracted"),
          py::arg("title"),
          py::arg("binary"));

    m.def("export_bezier_vtm", &io::export_bezier_vtm<double>,
          py::arg("path"),
          py::arg("patches"),
          py::arg("point_data"),
          py::arg("point_data_extracted"),
          py::arg("patch_names"),
          py::arg("title"),
          py::arg("binary"));

    m.def("bezier_anchor_params", &io::bezier_anchor_params<double>,
          py::arg("patch"));
}

}
