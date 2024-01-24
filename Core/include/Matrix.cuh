#pragma once

#include "CudaData.cuh"
#include "DeviceManager.cuh"
#include "MatrixOp.cuh"

template <typename T>
class Matrix : public CudaData<T>
{
public:
    Matrix(size_t nrow, size_t ncol, cudaStream_t stream = 0);
    Matrix(const T* hmem, size_t nrow, size_t ncol, cudaStream_t stream = 0);
    Matrix(const std::vector<T>& hmem, size_t nrow, size_t ncol,
           cudaStream_t stream = 0);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
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
    Matrix Transpose() const;

private:
    size_t m_nrow;
    size_t m_ncol;
};

template <typename T>
inline Matrix<T>::Matrix(size_t nrow, size_t ncol, cudaStream_t stream)
    : m_nrow(nrow), m_ncol(ncol), CudaData<T>(sizeof(T) * nrow * ncol, stream)
{
}

template <typename T>
inline Matrix<T>::Matrix(const T* hmem, size_t nrow, size_t ncol,
                         cudaStream_t stream)
    : m_nrow(nrow)
    , m_ncol(ncol)
    , CudaData<T>(hmem, sizeof(T) * nrow * ncol, stream)
{
}

template <typename T>
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

template <typename T>
inline Matrix<T>::Matrix(const Matrix& other)
    : CudaData<T>(other), m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix copy ctor\n");
#endif
}

template <typename T>
inline Matrix<T>::Matrix(Matrix&& other)
    : CudaData<T>(std::move(other)), m_nrow(other.m_nrow), m_ncol(other.m_ncol)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "Matrix move ctor\n");
#endif
}

template <typename T>
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

template <typename T>
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

template <typename T>
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

template <typename T>
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
    if (x.Shape() != y.Shape()) {
        fprintf(
            stdout,
            "Shape of x [%lu %lu] is not compatible with that of y [%lu %lu]"
            ", will return empty matrix\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
        return z;
    }

    auto sm_count = DeviceManager::Curr().Prop().multiProcessorCount;
    dim3 nb       = { (unsigned) sm_count, 4 }; // 120
    dim3 nt       = { 32, 8 };

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    MatrixOp::axpbyc<<<nb, nt, 0, s>>>(z.Data(), a, x.Data(), b, y.Data(), c,
                                       n_row, n_col);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Linear");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <typename T>
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
    if (x.Shape() != y.Shape()) {
        fprintf(
            stdout,
            "Shape of x [%lu %lu] is not compatible with that of y [%lu %lu]"
            ", will return empty matrix\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
        return z;
    }

    auto sm_count = DeviceManager::Curr().Prop().multiProcessorCount;
    dim3 nb       = { (unsigned) sm_count, 4 }; // 120
    dim3 nt       = { 32, 8 };

#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
#endif
    MatrixOp::cxamyb<<<nb, nt, 0, s>>>(z.Data(), c, x.Data(), a, y.Data(), b,
                                       n_row, n_col);
#ifdef DEBUG_PERFORMANCE
    Timer::Instance().Tick(s);
    Timer::Instance().ShowElapsedTime("Matrix Power");
#endif

    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(s));

    return z;
}

template <typename T>
inline Matrix<T> operator+(const Matrix<T>& x)
{
    return Linear((T) 1, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& x)
{
    return Linear((T) -1, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator+(T a, const Matrix<T>& x)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Matrix<T> operator+(const Matrix<T>& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Matrix<T> operator-(T a, const Matrix<T>& x)
{
    return Linear((T) -1, x, (T) 0, x, (T) a);
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& x, T a)
{
    return Linear((T) 1, x, (T) 0, x, (T) -a);
}

template <typename T>
inline Matrix<T> operator*(T a, const Matrix<T>& x)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator*(const Matrix<T>& x, T a)
{
    return Linear((T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator/(T a, const Matrix<T>& x)
{
    return Power((T) a, x, (T) -1, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator/(const Matrix<T>& x, T a)
{
    return Linear((T) 1 / (T) a, x, (T) 0, x, (T) 0);
}

template <typename T>
inline Matrix<T> operator+(const Matrix<T>& x, const Matrix<T>& y)
{
    return Linear((T) 1, x, (T) 1, y, (T) 0);
}

template <typename T>
inline Matrix<T> operator-(const Matrix<T>& x, const Matrix<T>& y)
{
    return Linear((T) 1, x, (T) -1, y, (T) 0);
}

template <typename T>
inline Matrix<T> operator*(const Matrix<T>& x, const Matrix<T>& y)
{
    return Power((T) 1, x, (T) 1, y, (T) 1);
}

template <typename T>
inline Matrix<T> operator/(const Matrix<T>& x, const Matrix<T>& y)
{
    return Power((T) 1, x, (T) 1, y, (T) -1);
}

template <typename T>
inline Matrix<T> MatMulSmall(const Matrix<T>& x, const Matrix<T>& y)
{
    cudaStream_t s = x.S();
    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(x.Nrow(), y.Ncol(), s);
    if (x.Ncol() != y.Nrow()) {
        fprintf(
            stdout,
            "Shape of x [%lu %lu] is not compatible with that of y [%lu %lu]"
            ", will return empty matrix\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
        return z;
    }

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

template <typename T>
inline Matrix<T> MatMulLarge(const Matrix<T>& x, const Matrix<T>& y)
{
    cudaStream_t s = x.S();
    if (x.S() != y.S()) {
        CUDA_CHECK(cudaDeviceSynchronize());
        s = 0;
    }

    Matrix<T> z(x.Nrow(), y.Ncol(), s);
    if (x.Ncol() != y.Nrow()) {
        fprintf(
            stdout,
            "Shape of x [%lu %lu] is not compatible with that of y [%lu %lu]"
            ", will return empty matrix\n",
            x.Shape()[0], x.Shape()[1], y.Shape()[0], y.Shape()[1]);
        return z;
    }

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

template <typename T>
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
