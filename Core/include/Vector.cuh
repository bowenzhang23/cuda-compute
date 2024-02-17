#pragma once

#include "CommonOp.cuh"
#include "CudaData.cuh"
#include "DeviceManager.cuh"
#include "VectorOp.cuh"
#include <iostream>

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
    T             Sum() const;
    T             Mean() const;
    ValueIndex<T> Max() const;
    ValueIndex<T> Min() const;
    Vector<T>     Reversed() const;
    void          Sort_(bool ascending = true);

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
inline T Vector<T>::Sum() const
{
    constexpr unsigned nt = 256;
    unsigned           nb = (Nlen() + 2 * nt - 1) / (2 * nt);
    Vector<T>          z(nb, this->S());

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    CommonOp::sum_unroll<T, nt>
        <<<nb, nt, 0, this->S()>>>(z.Data(), this->Data(), Nlen());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Vector Sum");
#endif
    auto sum_blocks = z.ToCPU();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->S()));

    return std::accumulate(sum_blocks.begin(), sum_blocks.end(), (T) 0);
}

template <NumericType T>
inline T Vector<T>::Mean() const
{
    return Sum() / (T) Nlen();
}

template <NumericType T>
inline ValueIndex<T> Vector<T>::Max() const
{
    constexpr unsigned    nt = 256;
    unsigned              nb = (Nlen() + 2 * nt - 1) / (2 * nt);
    Vector<ValueIndex<T>> z(nb, this->S());

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    CommonOp::max_unroll<T, nt>
        <<<nb, nt, 0, this->S()>>>(z.Data(), this->Data(), Nlen());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Vector Max");
#endif
    auto max_blocks = z.ToCPU();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->S()));

    auto max_iter = std::max_element(
        max_blocks.begin(), max_blocks.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.val < rhs.val; });
    return *max_iter;
}

template <NumericType T>
inline ValueIndex<T> Vector<T>::Min() const
{
    constexpr unsigned    nt = 256;
    unsigned              nb = (Nlen() + 2 * nt - 1) / (2 * nt);
    Vector<ValueIndex<T>> z(nb, this->S());

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    CommonOp::min_unroll<T, nt>
        <<<nb, nt, 0, this->S()>>>(z.Data(), this->Data(), Nlen());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Vector Min");
#endif
    auto min_blocks = z.ToCPU();
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->S()));

    auto min_iter = std::min_element(
        min_blocks.begin(), min_blocks.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.val < rhs.val; });
    return *min_iter;
}

template <NumericType T>
inline Vector<T> Vector<T>::Reversed() const
{
    constexpr unsigned nt       = 128;
    constexpr unsigned tile_dim = 512;
    unsigned           nb       = (Nlen() + tile_dim - 1) / tile_dim;
    Vector<T>          z(Nlen(), this->S());

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    VectorOp::reversed<T, tile_dim, nt>
        <<<nb, nt, 0, this->S()>>>(z.Data(), this->Data(), Nlen());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Vector Reversed");
#endif
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->S()));

    return z;
}

template <NumericType T>
inline void Vector<T>::Sort_(bool ascending)
{
    T    padding_value { 0 };
    auto n_next_po2 = VectorOp::next_po2(Nlen());

    constexpr unsigned nt = 128;
    unsigned           nb = (n_next_po2 + nt - 1) / nt;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    if (n_next_po2 == Nlen()) {
        for (unsigned k = 2; k <= n_next_po2; k <<= 1) {
            for (unsigned j = k >> 1; j > 0; j >>= 1) {
                VectorOp::sort<T><<<nb, nt, 0, this->S()>>>(this->Data(), k, j,
                                                            Nlen(), ascending);
            }
        }
    } else {
        padding_value      = ascending ? Max().val : Min().val;
        unsigned       len = n_next_po2 - Nlen();
        std::vector<T> padding_vec(len, padding_value);
        Vector<T>      padding(padding_vec, len, this->S());

        for (unsigned k = 2; k <= n_next_po2; k <<= 1) {
            for (unsigned j = k >> 1; j > 0; j >>= 1) {
                VectorOp::sort_padding<T><<<nb, nt, 0, this->S()>>>(
                    this->Data(), padding.Data(), k, j, Nlen(), ascending);
            }
        }
    }
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Vector Sort");
#endif
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->S()));
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

template <NumericType T>
inline T Mod2(const Vector<T>& x)
{
    return Inner(x, x);
}

template <NumericType T>
inline T Mod(const Vector<T>& x)
{
    return std::sqrt(Mod2(x));
}

template <NumericType T>
inline T Distance(const Vector<T>& x, const Vector<T>& y)
{
    if (&x == &y) return (T) 0;
    Vector<T> diff = x - y;
    return Mod(diff);
}
