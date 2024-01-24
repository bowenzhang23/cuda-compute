#include "Matrix.cuh"
#include "Vector.cuh"
#include "nanobind/nanobind.h"
#include "nanobind/stl/array.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;
using namespace nb::literals;

using Vectorf = Vector<float>;
using Vectori = Vector<int>;

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;

template <typename T, typename ContainerType, typename ElementType>
[[maybe_unused]] T& add_arithmetic(T& nb_class)
{
    return nb_class.def("__pos__", [](const ContainerType& a) { return a; })
        .def("__neg__", [](const ContainerType& a) { return -a; })
        .def("__pow__",
             [](const ContainerType& a, const ElementType b) {
                 return Power((ElementType) 1, a, b, a, (ElementType) 0);
             })
        .def("__add__",
             [](const ContainerType& a, const ElementType b) { return a + b; })
        .def("__radd__",
             [](const ContainerType& a, const ElementType b) { return b + a; })
        .def("__sub__",
             [](const ContainerType& a, const ElementType b) { return a - b; })
        .def("__rsub__",
             [](const ContainerType& a, const ElementType b) { return b - a; })
        .def("__mul__",
             [](const ContainerType& a, const ElementType b) { return a * b; })
        .def("__rmul__",
             [](const ContainerType& a, const ElementType b) { return b * a; })
        .def("__truediv__",
             [](const ContainerType& a, const ElementType b) { return a / b; })
        .def("__rtruediv__",
             [](const ContainerType& a, const ElementType b) { return b / a; })
        .def("__add__", [](const ContainerType& a,
                           const ContainerType& b) { return a + b; })
        .def("__sub__", [](const ContainerType& a,
                           const ContainerType& b) { return a - b; })
        .def("__mul__", [](const ContainerType& a,
                           const ContainerType& b) { return a * b; })
        .def("__truediv__", [](const ContainerType& a, const ContainerType& b) {
            return a / b;
        });
}

NB_MODULE(cuda_compute, m)
{
    auto vf = nb::class_<Vectorf>(m, "VectorfBase")
                  .def(nb::init<unsigned long>())
                  .def(nb::init<const std::vector<float>&, unsigned long>())
                  .def("cpu", &Vectorf::ToCPU)
                  .def("shape", &Vectorf::Shape);

    auto vi = nb::class_<Vectori>(m, "VectoriBase")
                  .def(nb::init<unsigned long>())
                  .def(nb::init<const std::vector<int>&, unsigned long>())
                  .def("cpu", &Vectori::ToCPU)
                  .def("shape", &Vectori::Shape);

    auto mf = nb::class_<Matrixf>(m, "MatrixfBase")
                  .def(nb::init<unsigned long, unsigned long>())
                  .def(nb::init<const std::vector<float>&, unsigned long,
                                unsigned long>())
                  .def("cpu", &Matrixf::ToCPU)
                  .def("shape", &Matrixf::Shape)
                  .def("transpose", &Matrixf::Transpose);

    auto mi = nb::class_<Matrixi>(m, "MatrixiBase")
                  .def(nb::init<unsigned long, unsigned long>())
                  .def(nb::init<const std::vector<int>&, unsigned long,
                                unsigned long>())
                  .def("cpu", &Matrixi::ToCPU)
                  .def("shape", &Matrixi::Shape)
                  .def("transpose", &Matrixi::Transpose);

    add_arithmetic<decltype(vf), Vectorf, float>(vf);
    add_arithmetic<decltype(vi), Vectori, int>(vi);
    add_arithmetic<decltype(mf), Matrixf, float>(mf);
    add_arithmetic<decltype(mi), Matrixi, int>(mi);

    m.def("device_query",
          []() { return DeviceManager::Instance().ToString(); });
    m.def("sgemm", &MatMul<float>, "a"_a, "b"_a);
    m.def("igemm", &MatMul<int>, "a"_a, "b"_a);
}
