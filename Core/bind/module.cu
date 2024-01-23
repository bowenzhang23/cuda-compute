#include "Matrix.cuh"
#include "Vector.cuh"
#include "nanobind/nanobind.h"
#include "nanobind/stl/array.h"
#include "nanobind/stl/vector.h"
#include "nanobind/stl/string.h"

namespace nb = nanobind;
using namespace nb::literals;

using Vectorf = Vector<float>;
using Vectori = Vector<int>;

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;

NB_MODULE(cuda_compute, m)
{
    nb::class_<Vectorf>(m, "VectorfBase")
        .def(nb::init<unsigned long>())
        .def(nb::init<const std::vector<float>&, unsigned long>())
        .def("cpu", &Vectorf::ToCPU)
        .def("shape", &Vectorf::Shape)
        .def("__mul__", [](const Vectorf& a, const float b) { return a * b; })
        .def("__rmul__", [](const Vectorf& a, const float b) { return a * b; })
        .def("__add__",
             [](const Vectorf& a, const Vectorf& b) { return a + b; })
        .def("__sub__",
             [](const Vectorf& a, const Vectorf& b) { return a - b; });

    nb::class_<Vectori>(m, "VectoriBase")
        .def(nb::init<unsigned long>())
        .def(nb::init<const std::vector<int>&, unsigned long>())
        .def("cpu", &Vectori::ToCPU)
        .def("shape", &Vectori::Shape)
        .def("__mul__", [](const Vectori& a, const int b) { return a * b; })
        .def("__rmul__", [](const Vectori& a, const int b) { return a * b; })
        .def("__add__",
             [](const Vectori& a, const Vectori& b) { return a + b; })
        .def("__sub__",
             [](const Vectori& a, const Vectori& b) { return a - b; });

    nb::class_<Matrixf>(m, "MatrixfBase")
        .def(nb::init<unsigned long, unsigned long>())
        .def(
            nb::init<const std::vector<float>&, unsigned long, unsigned long>())
        .def("cpu", &Matrixf::ToCPU)
        .def("shape", &Matrixf::Shape)
        .def("transpose", &Matrixf::Transpose)
        .def("__mul__", [](const Matrixf& a, const float b) { return a * b; })
        .def("__rmul__", [](const Matrixf& a, const float b) { return a * b; })
        .def("__add__",
             [](const Matrixf& a, const Matrixf& b) { return a + b; })
        .def("__sub__",
             [](const Matrixf& a, const Matrixf& b) { return a - b; });

    nb::class_<Matrixi>(m, "MatrixiBase")
        .def(nb::init<unsigned long, unsigned long>())
        .def(nb::init<const std::vector<int>&, unsigned long, unsigned long>())
        .def("cpu", &Matrixi::ToCPU)
        .def("shape", &Matrixi::Shape)
        .def("transpose", &Matrixi::Transpose)
        .def("__mul__", [](const Matrixi& a, const int b) { return a * b; })
        .def("__rmul__", [](const Matrixi& a, const int b) { return a * b; })
        .def("__add__",
             [](const Matrixi& a, const Matrixi& b) { return a + b; })
        .def("__sub__",
             [](const Matrixi& a, const Matrixi& b) { return a - b; });

    m.def("device_query", []() { return DeviceManager::Instance().ToString(); });
    m.def("sgemm", &MatMul<float>, "a"_a, "b"_a);
    m.def("igemm", &MatMul<int>, "a"_a, "b"_a);
}
