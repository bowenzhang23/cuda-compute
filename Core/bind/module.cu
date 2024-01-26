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
using Vectorb = Vector<bool>;

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;
using Matrixb = Matrix<bool>;

template <typename T, typename ContType, typename ElemType>
[[maybe_unused]] T& add_arithmetic(T& nb_class)
{
    return nb_class.def("__pos__", [](const ContType& a) { return a; })
        .def("__neg__", [](const ContType& a) { return -a; })
        .def("__pow__",
             [](const ContType& a, const ElemType b) {
                 return Power((ElemType) 1, a, b, a, (ElemType) 0);
             })
        .def("__add__",
             [](const ContType& a, const ElemType b) { return a + b; })
        .def("__radd__",
             [](const ContType& a, const ElemType b) { return b + a; })
        .def("__sub__",
             [](const ContType& a, const ElemType b) { return a - b; })
        .def("__rsub__",
             [](const ContType& a, const ElemType b) { return b - a; })
        .def("__mul__",
             [](const ContType& a, const ElemType b) { return a * b; })
        .def("__rmul__",
             [](const ContType& a, const ElemType b) { return b * a; })
        .def("__truediv__",
             [](const ContType& a, const ElemType b) { return a / b; })
        .def("__rtruediv__",
             [](const ContType& a, const ElemType b) { return b / a; })
        .def("__eq__",
             [](const ContType& a, const ElemType b) { return a == b; })
        .def("__req__",
             [](const ContType& a, const ElemType b) { return b == a; })
        .def("__ne__",
             [](const ContType& a, const ElemType b) { return a != b; })
        .def("__rne__",
             [](const ContType& a, const ElemType b) { return b != a; })
        .def("__gt__",
             [](const ContType& a, const ElemType b) { return a > b; })
        .def("__rgt__",
             [](const ContType& a, const ElemType b) { return b > a; })
        .def("__ge__",
             [](const ContType& a, const ElemType b) { return a >= b; })
        .def("__rge__",
             [](const ContType& a, const ElemType b) { return b >= a; })
        .def("__lt__",
             [](const ContType& a, const ElemType b) { return a < b; })
        .def("__rlt__",
             [](const ContType& a, const ElemType b) { return b < a; })
        .def("__le__",
             [](const ContType& a, const ElemType b) { return a <= b; })
        .def("__rle__",
             [](const ContType& a, const ElemType b) { return b <= a; })
        .def("__add__",
             [](const ContType& a, const ContType& b) { return a + b; })
        .def("__sub__",
             [](const ContType& a, const ContType& b) { return a - b; })
        .def("__mul__",
             [](const ContType& a, const ContType& b) { return a * b; })
        .def("__truediv__",
             [](const ContType& a, const ContType& b) { return a / b; })
        .def("__eq__",
             [](const ContType& a, const ContType& b) { return a == b; })
        .def("__ne__",
             [](const ContType& a, const ContType& b) { return a != b; })
        .def("__gt__",
             [](const ContType& a, const ContType& b) { return a > b; })
        .def("__ge__",
             [](const ContType& a, const ContType& b) { return a >= b; })
        .def("__lt__",
             [](const ContType& a, const ContType& b) { return a < b; })
        .def("__le__",
             [](const ContType& a, const ContType& b) { return a <= b; });
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
