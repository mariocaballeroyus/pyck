#include <pybind11/pybind11.h>

namespace pyck {

void bind_basis(pybind11::module_&);
void bind_geometry(pybind11::module_&);
void bind_quadrature(pybind11::module_&);
void bind_materials(pybind11::module_&);
void bind_elements(pybind11::module_&);
void bind_conditions(pybind11::module_&);
void bind_constraints(pybind11::module_&);
void bind_assembly(pybind11::module_&);
void bind_postprocessing(pybind11::module_&);
void bind_io(pybind11::module_&);

}

PYBIND11_MODULE(_pyck, m)
{
    pyck::bind_basis(m);
    pyck::bind_geometry(m);
    pyck::bind_quadrature(m);
    pyck::bind_materials(m);
    pyck::bind_elements(m);
    pyck::bind_conditions(m);
    pyck::bind_constraints(m);
    pyck::bind_assembly(m);
    pyck::bind_postprocessing(m);
    pyck::bind_io(m);
}
