#pragma once

#include "CommonOp.cuh"
#include "CudaData.cuh"
#include "DeviceManager.cuh"
#include "VectorOp.cuh"

template <NumericType T>
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

public:
    using value_type = T;
    using int_type   = Vector<int>;

private:
    size_t m_len;
};

template <NumericType T>
inline Vector<T>::Vector(size_t len, cudaStream_t stream)
    : m_len(len), CudaData<T>(sizeof(T) * len, stream)
{
}

template <NumericType T>
inline Vector<T>::Vector(const T* hmem, size_t len, cudaStream_t stream)
    : m_len(len), CudaData<T>(hmem, sizeof(T) * len, stream)
{
}

template <NumericType T>
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

template <NumericType T>
inline Vector<T>::Vector(const Vector& other)
    : CudaData<T>(other), m_len(other.m_len)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector copy ctor\n");
#endif
}

template <NumericType T>
inline Vector<T>::Vector(Vector&& other)
    : CudaData<T>(std::move(other)), m_len(other.m_len)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector move ctor\n");
#endif
}

template <NumericType T>
inline Vector<T>& Vector<T>::operator=(const Vector& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector copy assignment\n");
#endif
    CudaData<T>::operator=(other);
    this->m_len = other.m_len;
    return *this;
}

template <NumericType T>
inline Vector<T>& Vector<T>::operator=(Vector&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Vector move assignment\n");
#endif
    CudaData<T>::operator=(std::move(other));
    this->m_len = other.m_len;
    return *this;
}

template <NumericType T>
inline Vector<T> Linear(T a, const Vector<T>& x, T b, const Vector<T>& y, T c)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }
    Vector<T> z(len, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::axpbyc<<<nb, nt, 0, s>>>(z.Data(), a, x.Data(), b, y.Data(), c,
                                       len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Linear");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T>
inline Vector<T> Power(T c, const Vector<T>& x, T a, const Vector<T>& y, T b)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }
    Vector<T> z(len, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::cxamyb<<<nb, nt, 0, s>>>(z.Data(), c, x.Data(), a, y.Data(), b,
                                       len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T1, NumericType T2>
inline Vector<T1> Binary(const Vector<T2>& x, const Vector<T2>& y, BinaryOp op)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }
    Vector<T1> z(len, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::binary<T1, T2>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y.Data(), op, len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Binary");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T1, NumericType T2>
inline Vector<T1> Binary(const Vector<T2>& x, T2 y, BinaryOp op)
{
    cudaStream_t s = x.S();
    Vector<T1>   z(x.Nlen(), s);

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::binary<T1, T2>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y, op, x.Nlen());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T>
inline T Inner(const Vector<T>& x, const Vector<T>& y)
{
    cudaStream_t s   = x.S();
    auto         len = std::min(x.Nlen(), y.Nlen());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    if (!HasSameShape(x, y)) { return 0; }

    constexpr unsigned nt = 256;
    unsigned           nb = (len + 2 * nt - 1) / (2 * nt);
    Vector<T>          z(nb, s);

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    VectorOp::inner_unroll<T, nt>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y.Data(), len);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Vector Inner Product");
#endif
    auto inner_blocks = z.ToCPU();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return std::accumulate(inner_blocks.begin(), inner_blocks.end(), (T) 0);
}
