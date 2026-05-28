#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "linear_elastic_problem.hpp"
#include "element.hpp"
#include "quadrature.hpp"

namespace py = pybind11;

namespace pyck {

void bind_assembly(py::module_& m)
{
    using Patch1d = Patch<double, 1>;
    using Patch2d = Patch<double, 2>;
    using Element1d = Element<double, 1>;
    using Element2d = Element<double, 2>;
    using QuadratureRule1d = QuadratureRule<double, 1>;
    using QuadratureRule2d = QuadratureRule<double, 2>;

    // === LinearElasticProblem =======================================================

    using LinearElasticProblem1d = LinearElasticProblem<double, 1>;

    py::class_<LinearElasticProblem1d>(m, "LinearElasticProblem1d")

          // Constructor
          .def(py::init<const Ptr<Patch1d>&,
                        const Ptr<Element1d>&,
                        const Ptr<QuadratureRule1d>&>(),
               py::arg("patch"), 
               py::arg("element"), 
               py::arg("quadrature"))

          // Constructor
          .def(py::init<const std::vector<Ptr<Patch1d>>&,
                        const std::vector<Ptr<Element1d>>&,
                        const std::vector<Ptr<QuadratureRule1d>>&>(),
               py::arg("patches"), 
               py::arg("elements"), 
               py::arg("quadratures"))

          // Methods
          .def("add_patch", &LinearElasticProblem1d::add_patch,
               py::arg("patch"), 
               py::arg("element"), 
               py::arg("quadrature"))

          .def("add_condition", &LinearElasticProblem1d::add_condition,
               py::arg("condition"))

          .def("add_constraint", &LinearElasticProblem1d::add_direct_constraint,
               py::arg("constraint"))
          .def("add_constraint", &LinearElasticProblem1d::add_constraint,
               py::arg("constraint"))

          .def("assemble", [](const LinearElasticProblem1d& p) {
            Matrix<double> K;
            Vector<double> F;
            p.assemble(K, F);
            return py::make_tuple(K, F);
        });

    using LinearElasticProblem2d = LinearElasticProblem<double, 2>;
    py::class_<LinearElasticProblem2d>(m, "LinearElasticProblem2d")
        .def(py::init<>())
        .def(py::init<const Ptr<Patch2d>&,
                      const Ptr<Element2d>&,
                      const Ptr<QuadratureRule2d>&>(),
             py::arg("patch"), py::arg("element"), py::arg("quadrature"))
        .def(py::init<const std::vector<Ptr<Patch2d>>&,
                      const std::vector<Ptr<Element2d>>&,
                      const std::vector<Ptr<QuadratureRule2d>>&>(),
             py::arg("patches"), 
             py::arg("elements"), 
             py::arg("quadratures"))

        .def("add_patch", &LinearElasticProblem2d::add_patch,
             py::arg("patch"), 
             py::arg("element"), 
             py::arg("quadrature"))

        .def("add_condition", &LinearElasticProblem2d::add_condition,
             py::arg("condition"))

        .def("add_constraint", &LinearElasticProblem2d::add_direct_constraint,
             py::arg("constraint"))
        .def("add_constraint", &LinearElasticProblem2d::add_constraint,
             py::arg("constraint"))

        .def("assemble", [](const LinearElasticProblem2d& p) {
            Matrix<double> K;
            Vector<double> F;
            p.assemble(K, F);
            return py::make_tuple(K, F);
        });
}

}
