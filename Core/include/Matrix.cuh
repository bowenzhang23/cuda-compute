#pragma once

#include "CommonOp.cuh"
#include "CudaData.cuh"
#include "DeviceManager.cuh"
#include "MatrixOp.cuh"

template <NumericType T>
class Matrix : public CudaData<T>
{
public:
    Matrix(size_t nrow, size_t ncol, cudaStream_t stream = 0);
    Matrix(const T* hmem, size_t nrow, size_t ncol, cudaStream_t stream = 0);
    Matrix(const std::vector<T>& hmem, size_t nrow, size_t ncol,
           cudaStream_t stream = 0);
    Matrix(const Matrix& other);
    Matrix(const Vector<T>& other, size_t ncol);
    Matrix(Matrix&& other);
    Matrix(Vector<T>&& other, size_t ncol);
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other);

    inline size_t Nrow() const { return m_nrow; }
    inline size_t Ncol() const { return m_ncol; }

    virtual std::vector<T> ToCPU() const override
    {
        return CudaData<T>::ToCPU();
    }

    virtual inline std::vector<size_t> Shape() const override
    {
        return { Nrow(), Ncol() };
    }

public:
    using value_type = T;
    using int_type   = Matrix<int>;

public:
    void      Reshape_(size_t ncol);
    Matrix    Reshaped(size_t ncol);
    Vector<T> Into() const;
    Vector<T> Row(size_t i) const;
    Vector<T> Col(size_t i) const;

public:
    T             Sum() const;
    Vector<T>     Sum(uint8_t axis) const;
    T             Mean() const;
    Vector<T>     Mean(uint8_t axis) const;
    ValueIndex<T> Max() const;
    Vector<T>     Max(uint8_t axis) const;
    ValueIndex<T> Min() const;
    Vector<T>     Min(uint8_t axis) const;
    Matrix        Transpose() const;

private:
    size_t m_nrow;
    size_t m_ncol;
};

template <NumericType T>
inline Matrix<T>::Matrix(size_t nrow, size_t ncol, cudaStream_t stream)
    : m_nrow(nrow), m_ncol(ncol), CudaData<T>(sizeof(T) * nrow * ncol, stream)
{
}

template <NumericType T>
inline Matrix<T>::Matrix(const T* hmem, size_t nrow, size_t ncol,
                         cudaStream_t stream)
    : m_nrow(nrow)
    , m_ncol(ncol)
    , CudaData<T>(hmem, sizeof(T) * nrow * ncol, stream)
{
}

template <NumericType T>
inline Matrix<T>::Matrix(const std::vector<T>& hmem, size_t nrow, size_t ncol,
                         cudaStream_t stream)
    : m_nrow(nrow)
    , m_ncol(ncol)
    , CudaData<T>(hmem.data(), sizeof(T) * nrow * ncol, stream)
{
    if (nrow * ncol != hmem.size()) {
        fprintf(stdout,
                "Size of vector [%lu] != row x col [%lu]"
                ". Unexpected behaviour may occur!\n",
                hmem.size(), nrow * ncol);
    }
}

template <NumericType T>
inline Matrix<T>::Matrix(const Matrix& other)
    : CudaData<T>(other), m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix copy ctor\n");
#endif
}

template <NumericType T>
inline Matrix<T>::Matrix(const Vector<T>& other, size_t ncol)
    : CudaData<T>(other), m_nrow(other.Nlen() / ncol), m_ncol(ncol)
{
    if (Nrow() * Ncol() != other.Nlen()) {
        fprintf(stdout,
                "Size of vector [%lu] != row x col [%lu]"
                ". Unexpected behaviour may occur!\n",
                other.Nlen(), Nrow() * Ncol());
    }
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix copy Vector ctor\n");
#endif
}

template <NumericType T>
inline Matrix<T>::Matrix(Matrix&& other)
    : CudaData<T>(std::move(other)), m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix move ctor\n");
#endif
}

template <NumericType T>
inline Matrix<T>::Matrix(Vector<T>&& other, size_t ncol)
    : CudaData<T>(std::move(other)), m_nrow(other.Nlen() / ncol), m_ncol(ncol)
{
    if (Nrow() * Ncol() != other.Nlen()) {
        fprintf(stdout,
                "Size of vector [%lu] != row x col [%lu]"
                ". Unexpected behaviour may occur!\n",
                other.Nlen(), Nrow() * Ncol());
    }
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix move Vector ctor\n");
#endif
}

template <NumericType T>
inline Matrix<T>& Matrix<T>::operator=(const Matrix& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix copy assignment\n");
#endif
    CudaData<T>::operator=(other);
    this->m_nrow = other.m_nrow;
    this->m_ncol = other.m_ncol;
    return *this;
}

template <NumericType T>
inline Matrix<T>& Matrix<T>::operator=(Matrix&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix move assignment\n");
#endif
    CudaData<T>::operator=(std::move(other));
    this->m_nrow = other.m_nrow;
    this->m_ncol = other.m_ncol;
    return *this;
}

template <NumericType T>
inline void Matrix<T>::Reshape_(size_t ncol)
{
    auto nrow = m_nrow * m_ncol / ncol;
    if (m_nrow * m_ncol == nrow * ncol) {
        m_nrow = nrow;
        m_ncol = ncol;
    } else {
        fprintf(stdout, "Imcompatible size, failed to reshape!\n");
    }
}

template <NumericType T>
inline Matrix<T> Matrix<T>::Reshaped(size_t ncol)
{
    Matrix<T> new_mat(*this);
    new_mat.Reshape_(ncol);
    return new_mat;
}

template <NumericType T>
inline Vector<T> Matrix<T>::Into() const
{
    return Vector<T>(*this);
}

template <NumericType T>
inline Vector<T> Matrix<T>::Row(size_t i) const
{
    Vector<T> v(Ncol(), this->S());

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount;
    unsigned nt = 32;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    MatrixOp::cp_row<<<nb, nt, 0, this->S()>>>(v.Data(), this->Data(), i,
                                               Ncol());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Matrix Linear");
#endif

    return v;
}

template <NumericType T>
inline Vector<T> Matrix<T>::Col(size_t j) const
{
    Vector<T> v(Nrow(), this->S());

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount;
    unsigned nt = 32;
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    MatrixOp::cp_col<<<nb, nt, 0, this->S()>>>(v.Data(), this->Data(), j,
                                               Nrow(), Ncol());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Matrix Linear");
#endif

    return v;
}

template <NumericType T>
inline T Matrix<T>::Sum() const
{
    return Into().Sum();
}

template <NumericType T>
inline Vector<T> Matrix<T>::Sum(uint8_t axis) const
{
    std::vector<T> v;
    if (axis == 0) {
        for (size_t j = 0; j < Ncol(); ++j) { v.push_back(Col(j).Sum()); }
    } else {
        for (size_t i = 0; i < Nrow(); ++i) { v.push_back(Row(i).Sum()); }
    }
    return Vector<T>(v, v.size(), this->S());
}

template <NumericType T>
inline T Matrix<T>::Mean() const
{
    return Into().Mean();
}

template <NumericType T>
inline Vector<T> Matrix<T>::Mean(uint8_t axis) const
{
    std::vector<T> v;
    if (axis == 0) {
        for (size_t j = 0; j < Ncol(); ++j) { v.push_back(Col(j).Mean()); }
    } else {
        for (size_t i = 0; i < Nrow(); ++i) { v.push_back(Row(i).Mean()); }
    }
    return Vector<T>(v, v.size(), this->S());
}

template <NumericType T>
inline ValueIndex<T> Matrix<T>::Max() const
{
    return Into().Max();
}
template <NumericType T>
inline Vector<T> Matrix<T>::Max(uint8_t axis) const
{
    std::vector<T> v;
    if (axis == 0) {
        for (size_t j = 0; j < Ncol(); ++j) {
            // omitting idx
            v.push_back(Col(j).Max().val);
        }
    } else {
        for (size_t i = 0; i < Nrow(); ++i) { v.push_back(Row(i).Max().val); }
    }
    return Vector<T>(v, v.size(), this->S());
}

template <NumericType T>
inline ValueIndex<T> Matrix<T>::Min() const
{
    return Into().Min();
}
template <NumericType T>
inline Vector<T> Matrix<T>::Min(uint8_t axis) const
{
    std::vector<T> v;
    if (axis == 0) {
        for (size_t j = 0; j < Ncol(); ++j) {
            // omitting idx
            v.push_back(Col(j).Min().val);
        }
    } else {
        for (size_t i = 0; i < Nrow(); ++i) { v.push_back(Row(i).Min().val); }
    }
    return Vector<T>(v, v.size(), this->S());
}

template <NumericType T>
inline Matrix<T> Matrix<T>::Transpose() const
{
    Matrix<T> xt(Ncol(), Nrow(), this->S());

    constexpr unsigned tile_dim   = 32;
    constexpr unsigned block_rows = 8;

    dim3 nb = { ((unsigned) Nrow() + tile_dim - 1) / tile_dim,
                ((unsigned) Ncol() + tile_dim - 1) / tile_dim };
    dim3 nt = { tile_dim, block_rows };

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
#endif
    MatrixOp::transpose<<<nb, nt, 0, this->S()>>>(xt.Data(), this->Data(),
                                                  Nrow(), Ncol());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(this->S());
    Timer::Instance().ShowElapsedTime("Matrix Transpose");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(this->m_stream));
    return xt;
}

template <NumericType T>
inline Matrix<T> Linear(T a, const Matrix<T>& x, T b, const Matrix<T>& y, T c)
{
    cudaStream_t s     = x.S();
    auto         n_row = std::min(x.Nrow(), y.Nrow());
    auto         n_col = std::min(x.Ncol(), y.Ncol());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(n_row, n_col, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::axpbyc<<<nb, nt, 0, s>>>(z.Data(), a, x.Data(), b, y.Data(), c,
                                       n_row * n_col);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Linear");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T>
inline Matrix<T> Power(T c, const Matrix<T>& x, T a, const Matrix<T>& y, T b)
{
    cudaStream_t s     = x.S();
    auto         n_row = std::min(x.Nrow(), y.Nrow());
    auto         n_col = std::min(x.Ncol(), y.Ncol());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(n_row, n_col, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::cxamyb<<<nb, nt, 0, s>>>(z.Data(), c, x.Data(), a, y.Data(), b,
                                       n_row * n_col);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T1, NumericType T2, typename BinaryFunc>
inline Matrix<T1> Binary(const Matrix<T2>& x, const Matrix<T2>& y,
                         BinaryFunc func)
{
    cudaStream_t s     = x.S();
    auto         n_row = std::min(x.Nrow(), y.Nrow());
    auto         n_col = std::min(x.Ncol(), y.Ncol());

    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T1> z(n_row, n_col, s);
    if (!HasSameShape(x, y)) { return z; }

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::binary<T1, T2>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y.Data(), func, n_row * n_col);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T1, NumericType T2, typename BinaryFunc>
inline Matrix<T1> Binary(const Matrix<T2>& x, T2 y, BinaryFunc func)
{
    cudaStream_t s = x.S();
    Matrix<T1>   z(x.Nrow(), x.Ncol(), s);

    unsigned nb = DeviceManager::Curr().Prop().multiProcessorCount * 4;
    unsigned nt = 256;

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    CommonOp::binary<T1, T2>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y, func, x.Nrow() * x.Ncol());
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Binary");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <NumericType T>
inline Matrix<T> MatMulSmall(const Matrix<T>& x, const Matrix<T>& y)
{
    cudaStream_t s = x.S();
    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(x.Nrow(), y.Ncol(), s);
    if (!IsValidForGemm(x, y)) { return z; }

    unsigned m = (unsigned) x.Nrow();
    unsigned k = (unsigned) x.Ncol();
    unsigned n = (unsigned) y.Ncol();

    constexpr unsigned t = 32;

    dim3 nb = { (n + t - 1) / t, (m + t - 1) / t }; /* col, row */
    dim3 nt = { t * t };

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    MatrixOp::gemm_small<T, t>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y.Data(), m, k, n);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix MatMulSmall");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));
    return z;
}

template <NumericType T>
inline Matrix<T> MatMulLarge(const Matrix<T>& x, const Matrix<T>& y)
{
    cudaStream_t s = x.S();
    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(x.Nrow(), y.Ncol(), s);
    if (!IsValidForGemm(x, y)) { return z; }

    unsigned m = (unsigned) x.Nrow();
    unsigned k = (unsigned) x.Ncol();
    unsigned n = (unsigned) y.Ncol();

    constexpr unsigned tile_m  = 64;
    constexpr unsigned tile_k  = 8;
    constexpr unsigned tile_n  = 64;
    constexpr unsigned block_m = 8;
    constexpr unsigned block_n = 8;

    dim3 nb = { (n + tile_n - 1) / tile_n, (m + tile_m - 1) / tile_m };
    dim3 nt = { tile_m * tile_n / block_m / block_n };

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    MatrixOp::gemm_large<T, tile_m, tile_k, tile_n, block_m, block_n>
        <<<nb, nt, 0, s>>>(z.Data(), x.Data(), y.Data(), m, k, n);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix MatMulLarge");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));
    return z;
}

template <NumericType T>
inline Matrix<T> MatMul(const Matrix<T>& x, const Matrix<T>& y)
{
    unsigned m = (unsigned) x.Nrow();
    unsigned k = (unsigned) x.Ncol();
    unsigned n = (unsigned) y.Ncol();
    if (m > 512 && n > 512 && k > 256) {
        return MatMulLarge(x, y);
    } else {
        return MatMulSmall(x, y);
    }
}

template <NumericType T>
inline Vector<T> Inner(const Matrix<T>& x, const Vector<T>& y)
{
    return MatMul(x, y.Into(1));
}

template <NumericType T>
inline Vector<T> Inner(const Vector<T>& x, const Matrix<T>& y)
{
    return MatMul(x.Into(x.Nlen()), y);
}
