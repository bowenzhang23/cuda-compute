#include "Matrix.cuh"
#include "Vector.cuh"
#include "nanobind/nanobind.h"
#include "nanobind/stl/array.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

namespace nb = nanobind;
using namespace nb::literals;

using Vectorf = Vector<float>;
using Vectori = Vector<int>;

using Matrixf = Matrix<float>;
using Matrixi = Matrix<int>;

template <typename T>
std::pair<T, int> ToPair(const ValueIndex<T>& vi)
{
    return std::make_pair(vi.val, vi.idx);
}

template <typename Tnb, typename Tcd, typename Tcd_int, typename T>
[[maybe_unused]] Tnb& add_arithmetic(Tnb& cls)
{
    cls.def("__pos__", [](const Tcd& a) { return +a; });
    cls.def("__neg__", [](const Tcd& a) { return -a; });
    cls.def("__pow__", [](const Tcd& a, const T b) {
        return Power((T) 1, a, b, a, (T) 0);
    });
    cls.def("__add__", [](const Tcd& a, const T b) { return a + b; });
    cls.def("__radd__", [](const Tcd& a, const T b) { return b + a; });
    cls.def("__sub__", [](const Tcd& a, const T b) { return a - b; });
    cls.def("__rsub__", [](const Tcd& a, const T b) { return b - a; });
    cls.def("__mul__", [](const Tcd& a, const T b) { return a * b; });
    cls.def("__rmul__", [](const Tcd& a, const T b) { return b * a; });
    cls.def("__truediv__", [](const Tcd& a, const T b) { return a / b; });
    cls.def("__rtruediv__", [](const Tcd& a, const T b) { return b / a; });
    cls.def("__eq__", [](const Tcd& a, const T b) { return a == b; });
    cls.def("__req__", [](const Tcd& a, const T b) { return b == a; });
    cls.def("__ne__", [](const Tcd& a, const T b) { return a != b; });
    cls.def("__rne__", [](const Tcd& a, const T b) { return b != a; });
    cls.def("__gt__", [](const Tcd& a, const T b) { return a > b; });
    cls.def("__rgt__", [](const Tcd& a, const T b) { return b > a; });
    cls.def("__ge__", [](const Tcd& a, const T b) { return a >= b; });
    cls.def("__rge__", [](const Tcd& a, const T b) { return b >= a; });
    cls.def("__lt__", [](const Tcd& a, const T b) { return a < b; });
    cls.def("__rlt__", [](const Tcd& a, const T b) { return b < a; });
    cls.def("__le__", [](const Tcd& a, const T b) { return a <= b; });
    cls.def("__rle__", [](const Tcd& a, const T b) { return b <= a; });
    cls.def("__add__", [](const Tcd& a, const Tcd& b) { return a + b; });
    cls.def("__sub__", [](const Tcd& a, const Tcd& b) { return a - b; });
    cls.def("__mul__", [](const Tcd& a, const Tcd& b) { return a * b; });
    cls.def("__truediv__", [](const Tcd& a, const Tcd& b) { return a / b; });
    cls.def("__eq__", [](const Tcd& a, const Tcd& b) { return a == b; });
    cls.def("__ne__", [](const Tcd& a, const Tcd& b) { return a != b; });
    cls.def("__gt__", [](const Tcd& a, const Tcd& b) { return a > b; });
    cls.def("__ge__", [](const Tcd& a, const Tcd& b) { return a >= b; });
    cls.def("__lt__", [](const Tcd& a, const Tcd& b) { return a < b; });
    cls.def("__le__", [](const Tcd& a, const Tcd& b) { return a <= b; });

    return cls;
}

template <typename Tnb, typename Tcd, typename T>
[[maybe_unused]] Tnb& add_binary(Tnb& m)
{
    m.def("max", [](const Tcd& a, const T b) { return max(a, b); });
    m.def("max", [](const T b, const Tcd& a) { return max(b, a); });
    m.def("max", [](const Tcd& a, const Tcd& b) { return max(a, b); });
    m.def("min", [](const Tcd& a, const T b) { return min(a, b); });
    m.def("min", [](const T b, const Tcd& a) { return min(b, a); });
    m.def("min", [](const Tcd& a, const Tcd& b) { return min(a, b); });

    return m;
}

template <typename Tnb, typename Tcd, typename T>
[[maybe_unused]] Tnb& add_vector_common(Tnb& cls)
{
    cls.def(nb::init<unsigned long>());
    cls.def(nb::init<const std::vector<T>&, unsigned long>());
    cls.def("into", &Tcd::Into, "ncol"_a);
    cls.def("cpu", &Tcd::ToCPU);
    cls.def("shape", &Tcd::Shape);
    cls.def("sum", &Tcd::Sum);
    cls.def("mean", &Tcd::Mean);
    cls.def("max", [](const Tcd& a) { return ToPair(a.Max()); });
    cls.def("min", [](const Tcd& a) { return ToPair(a.Min()); });
    cls.def("reversed", &Tcd::Reversed);
    cls.def("sort_", &Tcd::Sort_, "ascending"_a = true);

    return cls;
}

template <typename Tnb, typename Tcd, typename T>
[[maybe_unused]] Tnb& add_matrix_common(Tnb& cls)
{
    cls.def(nb::init<unsigned long, unsigned long>());
    cls.def(nb::init<const std::vector<T>&, unsigned long, unsigned long>());
    cls.def("reshape_", &Tcd::Reshape_, "ncol"_a);
    cls.def("into", &Tcd::Into);
    cls.def("row", &Tcd::Row, "i"_a);
    cls.def("col", &Tcd::Col, "j"_a);
    cls.def("cpu", &Tcd::ToCPU);
    cls.def("shape", &Tcd::Shape);
    cls.def("transpose", &Tcd::Transpose);

    return cls;
}

template <typename Tnb, typename Tcd1, typename Tcd2, typename T>
[[maybe_unused]] Tnb& add_inner_product(Tnb& m)
{
    m.def(
        "inner", [](const Tcd1& a, const Tcd2& b) { return Inner<T>(a, b); },
        "a"_a, "b"_a);

    return m;
}

NB_MODULE(cuda_compute, m)
{
    nb::class_<Device>(m, "DeviceBase")
        .def(nb::init<int>())
        .def("id", &Device::ID);

    auto vf = nb::class_<Vectorf>(m, "VectorfBase");
    auto vi = nb::class_<Vectori>(m, "VectoriBase");

    auto mf = nb::class_<Matrixf>(m, "MatrixfBase");
    auto mi = nb::class_<Matrixi>(m, "MatrixiBase");

    add_arithmetic<decltype(vf), Vectorf, Vectori, float>(vf);
    add_arithmetic<decltype(vi), Vectori, Vectori, int>(vi);
    add_arithmetic<decltype(mf), Matrixf, Matrixi, float>(mf);
    add_arithmetic<decltype(mi), Matrixi, Matrixi, int>(mi);

    add_vector_common<decltype(vf), Vectorf, float>(vf);
    add_vector_common<decltype(vi), Vectori, int>(vi);
    add_matrix_common<decltype(mf), Matrixf, float>(mf);
    add_matrix_common<decltype(mi), Matrixi, int>(mi);

    add_binary<decltype(m), Vectorf, float>(m);
    add_binary<decltype(m), Vectori, int>(m);
    add_binary<decltype(m), Matrixf, float>(m);
    add_binary<decltype(m), Matrixi, int>(m);

    add_inner_product<decltype(m), Vectorf, Vectorf, float>(m);
    add_inner_product<decltype(m), Vectori, Vectori, int>(m);
    add_inner_product<decltype(m), Matrixf, Vectorf, float>(m);
    add_inner_product<decltype(m), Matrixi, Vectori, int>(m);
    add_inner_product<decltype(m), Vectorf, Matrixf, float>(m);
    add_inner_product<decltype(m), Vectori, Matrixi, int>(m);

    m.def("use_device",
          [](int id) { DeviceManager::Instance().UseDevice(id); });
    m.def("current_device",
          []() { return DeviceManager::Instance().CurrentDevice(); });
    m.def("device_query",
          []() { return DeviceManager::Instance().ToString(); });
    m.def("timer_start", []() { Timer::Instance().Tick(); });
    m.def("timer_stop", []() {
        Timer::Instance().Tick();
        return Timer::Instance().ElapsedTime();
    });

    m.def("gemm", &MatMul<float>, "a"_a, "b"_a);
    m.def("gemm", &MatMul<int>, "a"_a, "b"_a);
    m.def("mod2", &Mod2<float>, "a"_a);
    m.def("mod", &Mod<float>, "a"_a);
    m.def("distance", &Distance<float>, "a"_a, "b"_a);
}
