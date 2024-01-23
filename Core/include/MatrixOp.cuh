#pragma once

#include "cuda_runtime.h"

namespace MatrixOp
{
template <typename T>
__global__ void axpby(T* Z, const T a, const T* X, const T b, const T* Y,
                      unsigned n_row, unsigned n_col)
{
    auto col = threadIdx.x + blockIdx.x * blockDim.x;
    auto row = threadIdx.y + blockIdx.y * blockDim.y;

    auto stride_col = blockDim.x * gridDim.x;
    auto stride_row = blockDim.y * gridDim.y;

    for (; col < n_col; col += stride_col) {
        for (; row < n_row; row += stride_row) {
            Z[row * n_col + col] =
                a * X[row * n_col + col] + b * Y[row * n_col + col];
        }
    }
}

template <typename T, unsigned tile_dim = 32, unsigned block_rows = 8>
__global__ void transpose(T* XT, const T* X, unsigned n_row, unsigned n_col)
{
    __shared__ T tmp[tile_dim][tile_dim + 1];

    auto col = threadIdx.x + blockIdx.x * tile_dim;
    auto row = threadIdx.y + blockIdx.y * tile_dim;

    for (int j = 0; j < tile_dim && row + j < n_row; j += block_rows) {
        tmp[threadIdx.y + j][threadIdx.x] = X[(row + j) * n_col + col];
    }

    __syncthreads();

    col = threadIdx.x + blockIdx.y * tile_dim;
    row = threadIdx.y + blockIdx.x * tile_dim;

    for (int j = 0; j < tile_dim && row + j < n_col; j += block_rows) {
        XT[(row + j) * n_row + col] = tmp[threadIdx.x][threadIdx.y + j];
    }
}

template <typename T, unsigned tile_dim = 32>
__global__ void gemm_small(T* C, const T* A, const T* B, unsigned m, unsigned k,
                           unsigned n)
{
    // Dimensions
    //  A: m x k
    //  B: k x n
    //  C: m x n
    // w.r.t tile dimension of the entire matrix
    auto tile_row = blockIdx.y;
    auto tile_col = blockIdx.x;

    __shared__ T As[tile_dim][tile_dim];
    __shared__ T Bs[tile_dim][tile_dim];

    // w.r.t thread dimension of one tile
    int thread_row = threadIdx.x / tile_dim;
    int thread_col = threadIdx.x % tile_dim;

    int idx_a = tile_row * tile_dim * k;
    int idx_b = tile_col * tile_dim;

    T sum { 0 };
    // loop over tiles
    for (int itile = 0; itile < k; itile += tile_dim) {
        int pos_a = idx_a + thread_row * k + thread_col;
        int pos_b = idx_b + thread_row * n + thread_col;

        As[thread_row][thread_col] = pos_a < m * k ? A[pos_a] : 0;
        Bs[thread_row][thread_col] = pos_b < k * n ? B[pos_b] : 0;
        __syncthreads();

        idx_a += tile_dim;
        idx_b += tile_dim * n;

        // loop over elements inside tile
        for (int j = 0; j < tile_dim; ++j) {
            sum += As[thread_row][j] * Bs[j][thread_col];
        }
        __syncthreads();
    }
    int row = tile_row * tile_dim + thread_row;
    int col = tile_col * tile_dim + thread_col;

    if (row < m && col < n) { C[row * n + col] = sum; }
}

template <typename T, unsigned tile_m = 64, unsigned tile_k = 8,
          unsigned tile_n = 64, unsigned block_m = 8, unsigned block_n = 8>
__global__ void gemm_large(T* C, const T* A, const T* B, unsigned m, unsigned k,
                           unsigned n)
{
    // Dimensions
    //  A: m x k
    //  B: k x n
    //  C: m x n
    // w.r.t tile dimension of the entire matrix
    auto tile_row = blockIdx.y;
    auto tile_col = blockIdx.x;

    __shared__ T As[tile_k][tile_m];
    __shared__ T Bs[tile_k][tile_n];

    // w.r.t thread dimension of one tile
    int thread_row = threadIdx.x / (tile_n / block_n);
    int thread_col = threadIdx.x % (tile_n / block_n);

    int idx_a = tile_row * tile_m * k;
    int idx_b = tile_col * tile_n;

    // w.r.t the inner dimension inside a thread
    int irow_a   = threadIdx.x / (tile_k);
    int icol_a   = threadIdx.x % (tile_k);
    int irow_b   = threadIdx.x / (tile_n);
    int icol_b   = threadIdx.x % (tile_n);
    int stride_a = blockDim.x / tile_k;
    int stride_b = blockDim.x / tile_n;

    T sum[block_m * block_n] = { 0 };
    T tmp_a[block_m]         = { 0 };
    T tmp_b[block_n]         = { 0 };

    // loop over tiles
    for (int itile = 0; itile < k; itile += tile_k) {
        for (int offset = 0; offset < tile_m; offset += stride_a) {
            int pos_a = idx_a + (irow_a + offset) * k + icol_a;
            As[icol_a][irow_a + offset] = pos_a < m * k ? A[pos_a] : 0;
        }
        for (int offset = 0; offset < tile_k; offset += stride_b) {
            int pos_b = idx_b + (irow_b + offset) * n + icol_b;
            Bs[irow_b + offset][icol_b] = pos_b < k * n ? B[pos_b] : 0;
        }
        __syncthreads();

        idx_a += tile_k;
        idx_b += tile_k * n;

        // loop over blocks inside tile
        for (int iblock = 0; iblock < tile_k; ++iblock) {
            // loop over elements in block
            for (int i = 0; i < block_m; ++i) {
                tmp_a[i] = As[iblock][thread_row * block_m + i];
            }
            for (int i = 0; i < block_n; ++i) {
                tmp_b[i] = Bs[iblock][thread_col * block_n + i];
            }
            for (int im = 0; im < block_m; ++im) {
                for (int in = 0; in < block_n; ++in) {
                    sum[im * block_n + in] += tmp_a[im] * tmp_b[in];
                }
            }
        }
        __syncthreads();
    }

    int row = tile_row * tile_m + thread_row * block_m;
    int col = tile_col * tile_n + thread_col * block_n;

    for (int im = 0; im < block_m && row + im < m; ++im) {
        for (int in = 0; in < block_n && col + in < n; ++in) {
            C[(row + im) * n + (col + in)] = sum[im * block_n + in];
        }
    }
}

} // namespace MatrixOp