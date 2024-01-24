#pragma once

#include "CudaData.cuh"
#include "DeviceManager.cuh"
#include "VectorOp.cuh"

template <typename T>
class Vector : public CudaData<T>
{
public:
    Vector(size_t len, cudaStream_t stream = 0);
    Vector(const T* hmem, size_t len, cudaStream_t stream = 0);
    Vector(const std::vector<T>& hmem, size_t len, cudaStream_t stream = 0);
    Vector(const Vector& other);
    Vector(Vector&& other);
    Vector& operator=(const Vector& other);
    Vector& operator=(Vector&& other);

    inline size_t Nlen() const { return this->m_len; }

    virtual std::vector<T> ToCPU() const override
    {
        return CudaData<T>::ToCPU();
    }

    virtual inline std::vector<size_t> Shape() const override
    {
        return { Nlen() };
    }

private:
    size_t m_len;
};

template <typename T>
inline Vector<T>::Vector(size_t len, cudaStream_t stream)
    : m_len(len), CudaData<T>(sizeof(T) * len, stream)
{
}

template <typename T>
inline Vector<T>::Vector(const T* hmem, size_t len, cudaStream_t stream)
    : m_len(len), CudaData<T>(hmem, sizeof(T) * len, stream)
{
}

template <typename T>
inline Vector<T>::Vector(const std::vector<T>& hmem, size_t len,
                         cudaStream_t stream)
    : m_len(len), CudaData<T>(hmem.data(), sizeof(T) * len, stream)
{
    if (len != hmem.size()) {
        fprintf(stdout,
                "Size of vector [%lu] != len [%lu]"
                ". Unexpected behaviour may occur!\n",
                hmem.size(), len);
    }
}

template <typename T>
inline Vector<T>::Vector(const Vector& other)
    : CudaData<T>(other), m_len(other.m_len)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector copy ctor\n");
#endif
}

template <typename T>
inline Vector<T>::Vector(Vector&& other)
    : CudaData<T>(std::move(other)), m_len(other.m_len)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector move ctor\n");
#endif
}

template <typename T>
inline Vector<T>& Vector<T>::operator=(const Vector& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector copy assignment\n");
#endif
    CudaData<T>::operator=(other);
    this->m_len = other.m_len;
    return *this;
}

template <typename T>
inline Vector<T>& Vector<T>::operator=(Vector&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector move assignment\n");
#endif
    CudaData<T>::operator=(std::move(other));
    this->m_len = other.m_len;
    return *this;
}

template <typename T>
inline Vector<T> Linear(T a, const Vector<T>& x, T b, const Vector<T>& y, T c)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }
    Vector<T> z(len, s);
    if (x.Shape() != y.Shape()) {
        fprintf(stdout,
                "Shape of x [%lu] is not compatible with that of y [%lu]"
                ", will return empty matrix\n",
                x.Shape()[0], y.Shape()[0]);
        return z;
    }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4; // 120
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    VectorOp::axpbyc<<<nb, nt, 0, s>>>(z.Data(), a, x.Data(), b, y.Data(), c,
                                       len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Linear");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <typename T>
inline Vector<T> Power(T c, const Vector<T>& x, T a, const Vector<T>& y, T b)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }
    Vector<T> z(len, s);
    if (x.Shape() != y.Shape()) {
        fprintf(stdout,
                "Shape of x [%lu] is not compatible with that of y [%lu]"
                ", will return empty matrix\n",
                x.Shape()[0], y.Shape()[0]);
        return z;
    }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4; // 120
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    VectorOp::cxamyb<<<nb, nt, 0, s>>>(z.Data(), c, x.Data(), a, y.Data(), b,
                                       len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <typename T>
inline Vector<T> operator+(const Vector<T>& x)
{
    return Linear((T) 1, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& x)
{
    return Linear((T) -1, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Vector<T> operator+(T a, const Vector<T>& x)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Vector<T> operator+(const Vector<T>& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Vector<T> operator-(T a, const Vector<T>& x)
{
    return Linear((T) -1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) -a);
}

template <typename T>
inline Vector<T> operator*(T a, const Vector<T>& x)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Vector<T> operator*(const Vector<T>& x, T a)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Vector<T> operator/(T a, const Vector<T>& x)
{
    return Power((T) a, x, (T) -1, x, (T) 0);
}

template <typename T>
inline Vector<T> operator/(const Vector<T>& x, T a)
{
    return Linear((T) 1 / (T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Vector<T> operator+(const Vector<T>& x, const Vector<T>& y)
{
    return Linear((T) 1, x, (T) 1, y, (T) 0);
}

template <typename T>
inline Vector<T> operator-(const Vector<T>& x, const Vector<T>& y)
{
    return Linear((T) 1, x, (T) -1, y, (T) 0);
}

template <typename T>
inline Vector<T> operator*(const Vector<T>& x, const Vector<T>& y)
{
    return Power((T) 1, x, (T) 1, y, (T) 1);
}

template <typename T>
inline Vector<T> operator/(const Vector<T>& x, const Vector<T>& y)
{
    return Power((T) 1, x, (T) 1, y, (T) -1);
}
