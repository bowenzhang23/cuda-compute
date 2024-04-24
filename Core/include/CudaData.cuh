#pragma once

#include "Error.cuh"
#include "Timer.cuh"
#include "cuda_runtime.h"
#include <concepts>
#include <numeric>
#include <vector>

// #define DEBUG_CONSTRUCTOR
// #define DEBUG_PERFORMANCE

template <typename T>
concept NumericType = std::is_standard_layout_v<T>;

template <NumericType T>
class Vector;
template <NumericType T>
class Matrix;
template <NumericType T>
class CudaData;

template <NumericType T>
class CudaData
{
public:
    CudaData(size_t size, cudaStream_t stream = 0);
    CudaData(const T* hmem, size_t size, cudaStream_t stream = 0);
    CudaData(const CudaData& other);
    CudaData(CudaData&& other);
    CudaData& operator=(const CudaData& other);
    CudaData& operator=(CudaData&& other);

    virtual ~CudaData();

    virtual std::vector<T>             ToCPU() const;
    virtual inline std::vector<size_t> Shape() const { return {}; }

    inline T*                  Data() { return m_dmem; }
    inline const T*            Data() const { return m_dmem; }
    inline const cudaStream_t& S() const { return m_stream; }

public:
    using value_type = T;
    using int_type   = CudaData<int>;

protected:
    cudaStream_t m_stream;
    size_t       m_size;
    T*           m_dmem = nullptr;
    T*           m_hmem = nullptr;
};

template <NumericType T>
inline CudaData<T>::CudaData(size_t size, cudaStream_t stream)
    : m_size(size), m_stream(stream)
{
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    m_hmem = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    CUDA_CHECK(cudaMemsetAsync((void*) m_dmem, 0, m_size, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Construct zeros");
#endif
}

template <NumericType T>
inline CudaData<T>::CudaData(const T* hmem, size_t size, cudaStream_t stream)
    : m_size(size), m_stream(stream)
{
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    m_hmem = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, m_hmem, m_size, cudaMemcpyHostToDevice,
                               m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Construct from data");
#endif
}

template <NumericType T>
inline CudaData<T>::CudaData(const CudaData& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData copy ctor\n");
#endif
    if (this == &other) return;
    m_size   = other.m_size;
    m_stream = other.m_stream;
    m_hmem   = (T*) calloc(1, m_size);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, other.m_hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, other.m_dmem, m_size,
                               cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Copy construct");
#endif
}

template <NumericType T>
inline CudaData<T>::CudaData(CudaData&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData move ctor\n");
#endif
    if (this == &other) return;
    m_size   = other.m_size;
    m_stream = other.m_stream;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    std::swap(m_hmem, other.m_hmem);
    std::swap(m_dmem, other.m_dmem);
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Move construct");
#endif
}

template <NumericType T>
inline CudaData<T>& CudaData<T>::operator=(const CudaData& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData copy assignment\n");
#endif
    if (this == &other) return *this;
    m_size   = other.m_size;
    m_stream = other.m_stream;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    m_hmem = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, other.m_hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, other.m_dmem, m_size,
                               cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Copy assignment");
#endif
    return *this;
}

template <NumericType T>
inline CudaData<T>& CudaData<T>::operator=(CudaData&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData move assignment\n");
#endif
    if (this == &other) return *this;
    m_size   = other.m_size;
    m_stream = other.m_stream;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    std::swap(m_hmem, other.m_hmem);
    std::swap(m_dmem, other.m_dmem);
    CUDA_CHECK(cudaStreamSynchronize(S()));
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("Move assignment");
#endif
    return *this;
}

template <NumericType T>
inline CudaData<T>::~CudaData()
{
    free(m_hmem);
    CUDA_CHECK(cudaFreeAsync((void*) m_dmem, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
}

template <NumericType T>
inline std::vector<T> CudaData<T>::ToCPU() const
{
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
#endif
    auto           count = m_size / sizeof(T);
    std::vector<T> hmem(count);
    CUDA_CHECK(cudaMemcpyAsync(m_hmem, m_dmem, m_size, cudaMemcpyDeviceToHost,
                               m_stream));
    CUDA_CHECK(cudaStreamSynchronize(S()));
    memcpy((void*) hmem.data(), m_hmem, m_size);
    return hmem;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(m_stream);
    Timer::Instance().ShowElapsedTime("To CPU");
#endif
}

template <NumericType T>
inline bool HasSameShape(const Vector<T>& x, const Vector<T>& y)
{
    if (x.Shape() == y.Shape()) return true;

    fprintf(stdout,
            "Shape of x [%lu] is not compatible with y [%lu]"
            ", will return empty\n",
            x.Shape()[0], y.Shape()[0]);
    return false;
}

template <NumericType T>
inline bool HasSameShape(const Matrix<T>& x, const Matrix<T>& y)
{
    if (x.Shape() == y.Shape()) return true;

    fprintf(stdout,
            "Shape of x [%lu %lu] is not compatible with y [%lu %lu]"
            ", will return empty\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
    return false;
}

template <NumericType T>
inline bool IsValidForGemm(const Matrix<T>& x, const Matrix<T>& y)
{
    if (x.Ncol() == y.Nrow()) return true;

    fprintf(stdout,
            "Shape of x [%lu %lu] is not compatible with y [%lu %lu]"
            ", will return empty\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
    return false;
}

template <typename Tcd>
inline Tcd operator+(const Tcd& x)
{
    using T = typename Tcd::value_type;
    return Linear((T) 1, x, (T) 0, x, (T) 0);
}

template <typename Tcd>
inline Tcd operator-(const Tcd& x)
{
    using T = typename Tcd::value_type;
    return Linear((T) -1, x, (T) 0, x, (T) 0);
}

template <NumericType T, typename Tcd>
inline Tcd operator+(T a, const Tcd& x)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <NumericType T, typename Tcd>
inline Tcd operator+(const Tcd& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <NumericType T, typename Tcd>
inline Tcd operator-(T a, const Tcd& x)
{
    return Linear((T) -1, x, (T) 0, x, (T) a);
}

template <NumericType T, typename Tcd>
inline Tcd operator-(const Tcd& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) -a);
}

template <NumericType T, typename Tcd>
inline Tcd operator*(T a, const Tcd& x)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <NumericType T, typename Tcd>
inline Tcd operator*(const Tcd& x, T a)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <NumericType T, typename Tcd>
inline Tcd operator/(T a, const Tcd& x)
{
    return Power((T) a, x, (T) -1, x, (T) 0);
}

template <NumericType T, typename Tcd>
inline Tcd operator/(const Tcd& x, T a)
{
    return Linear((T) 1 / (T) a, x, (T) 0, x, (T) 0);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator==(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::EQ);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator==(T a, const Tcd& x)
{
    return x == a;
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator!=(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::NE);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator!=(T a, const Tcd& x)
{
    return x != a;
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator<(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::LT);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator<(T a, const Tcd& x)
{
    return x > a;
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator<=(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::LE);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator<=(T a, const Tcd& x)
{
    return x >= a;
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator>(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::GT);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator>(T a, const Tcd& x)
{
    return x < a;
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator>=(const Tcd& x, T a)
{
    return Binary<int, T>(x, a, BinaryOp::GE);
}

template <NumericType T, typename Tcd>
inline typename Tcd::int_type operator>=(T a, const Tcd& x)
{
    return x <= a;
}

template <NumericType T, typename Tcd>
inline Tcd max(const Tcd& x, T a)
{
    return Binary<T, T>(x, a, BinaryOp::MAX);
}

template <NumericType T, typename Tcd>
inline Tcd max(T a, const Tcd& x)
{
    return max(x, a);
}

template <NumericType T, typename Tcd>
inline Tcd min(const Tcd& x, T a)
{
    return Binary<T, T>(x, a, BinaryOp::MIN);
}

template <NumericType T, typename Tcd>
inline Tcd min(T a, const Tcd& x)
{
    return min(x, a);
}

template <typename Tcd>
inline Tcd operator+(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Linear((T) 1, x, (T) 1, y, (T) 0);
}

template <typename Tcd>
inline Tcd operator-(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Linear((T) 1, x, (T) -1, y, (T) 0);
}

template <typename Tcd>
inline Tcd operator*(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Power((T) 1, x, (T) 1, y, (T) 1);
}

template <typename Tcd>
inline Tcd operator/(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Power((T) 1, x, (T) 1, y, (T) -1);
}

template <typename Tcd>
inline typename Tcd::int_type operator==(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::EQ);
}

template <typename Tcd>
inline typename Tcd::int_type operator!=(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::NE);
}

template <typename Tcd>
inline typename Tcd::int_type operator<(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::LT);
}

template <typename Tcd>
inline typename Tcd::int_type operator<=(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::LE);
}

template <typename Tcd>
inline typename Tcd::int_type operator>(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::GT);
}

template <typename Tcd>
inline typename Tcd::int_type operator>=(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<int, T>(x, y, BinaryOp::GE);
}

template <typename Tcd>
inline Tcd max(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<T, T>(x, y, BinaryOp::MAX);
}

template <typename Tcd>
inline Tcd min(const Tcd& x, const Tcd& y)
{
    using T = typename Tcd::value_type;
    return Binary<T, T>(x, y, BinaryOp::MIN);
}