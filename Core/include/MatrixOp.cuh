#pragma once

#include "cuda_runtime.h"

namespace MatrixOp
{
template <typename T, unsigned tile_dim = 32, unsigned block_rows = 8,
          unsigned padding = 1>
__global__ void transpose(T* XT, const T* X, unsigned n_row, unsigned n_col)
{
    __shared__ T tmp[tile_dim][tile_dim + padding];

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

template <typename T>
[[deprecated("Replaced by transpose using smem.")]] __global__ void
transpose_diag_coord(T* XT, const T* X, unsigned n_row, unsigned n_col)
{
    auto blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    auto blk_y = blockIdx.x;
    auto row   = blockDim.x * blk_x + threadIdx.x;
    auto col   = blockDim.y * blk_y + threadIdx.y;

    if (row < n_row && col < n_col) {
        XT[col * n_row + row] = X[row * n_col + col];
    }
}

template <typename T, unsigned tile_dim = 32>
__global__ void gemm_small(T* Z, const T* X, const T* Y, unsigned m, unsigned k,
                           unsigned n)
{
    // Dimensions
    //  X: m x k
    //  Y: k x n
    //  Z: m x n
    // w.r.t tile dimension of the entire matrix
    auto tile_row = blockIdx.y;
    auto tile_col = blockIdx.x;

    __shared__ T Xs[tile_dim][tile_dim];
    __shared__ T Ys[tile_dim][tile_dim];

    // w.r.t thread dimension of one tile
    int thread_row = threadIdx.x / tile_dim;
    int thread_col = threadIdx.x % tile_dim;

    int idx_x = tile_row * tile_dim * k;
    int idx_y = tile_col * tile_dim;

    T sum { 0 };
    // loop over tiles
    for (int itile = 0; itile < k; itile += tile_dim) {
        int pos_x = idx_x + thread_row * k + thread_col;
        int pos_y = idx_y + thread_row * n + thread_col;

        Xs[thread_row][thread_col] = pos_x < m * k ? X[pos_x] : 0;
        Ys[thread_row][thread_col] = pos_y < k * n ? Y[pos_y] : 0;
        __syncthreads();

        idx_x += tile_dim;
        idx_y += tile_dim * n;

        // loop over elements inside tile
        for (int j = 0; j < tile_dim; ++j) {
            sum += Xs[thread_row][j] * Ys[j][thread_col];
        }
        __syncthreads();
    }
    int row = tile_row * tile_dim + thread_row;
    int col = tile_col * tile_dim + thread_col;

    if (row < m && col < n) { Z[row * n + col] = sum; }
}

template <typename T, unsigned tile_m = 64, unsigned tile_k = 8,
          unsigned tile_n = 64, unsigned block_m = 8, unsigned block_n = 8,
          unsigned padding = 4>
__global__ void gemm_large(T* Z, const T* X, const T* Y, unsigned m, unsigned k,
                           unsigned n)
{
    // Dimensions
    //  X: m x k
    //  Y: k x n
    //  Z: m x n
    // w.r.t tile dimension of the entire matrix
    auto tile_row = blockIdx.y;
    auto tile_col = blockIdx.x;

    __shared__ T Xs[tile_k][tile_m + padding];
    __shared__ T Ys[tile_k][tile_n + padding];

    // w.r.t thread dimension of one tile
    int thread_row = threadIdx.x / (tile_n / block_n);
    int thread_col = threadIdx.x % (tile_n / block_n);

    int idx_x = tile_row * tile_m * k;
    int idx_y = tile_col * tile_n;

    // w.r.t the inner dimension inside a thread
    int irow_x   = threadIdx.x / (tile_k);
    int icol_x   = threadIdx.x % (tile_k);
    int irow_y   = threadIdx.x / (tile_n);
    int icol_y   = threadIdx.x % (tile_n);
    int stride_x = blockDim.x / tile_k;
    int stride_y = blockDim.x / tile_n;

    T sum[block_m * block_n] = { 0 };
    T tmp_x[block_m]         = { 0 };
    T tmp_y[block_n]         = { 0 };

    // loop over tiles
    for (int itile = 0; itile < k; itile += tile_k) {
        for (int offset = 0; offset < tile_m; offset += stride_x) {
            int pos_x = idx_x + (irow_x + offset) * k + icol_x;
            Xs[icol_x][irow_x + offset] = pos_x < m * k ? X[pos_x] : 0;
        }
        for (int offset = 0; offset < tile_k; offset += stride_y) {
            int pos_y = idx_y + (irow_y + offset) * n + icol_y;
            Ys[irow_y + offset][icol_y] = pos_y < k * n ? Y[pos_y] : 0;
        }
        __syncthreads();

        idx_x += tile_k;
        idx_y += tile_k * n;

        // loop over blocks inside tile
        for (int iblock = 0; iblock < tile_k; ++iblock) {
            // loop over elements in block
            for (int i = 0; i < block_m; ++i) {
                tmp_x[i] = Xs[iblock][thread_row * block_m + i];
            }
            for (int i = 0; i < block_n; ++i) {
                tmp_y[i] = Ys[iblock][thread_col * block_n + i];
            }
            for (int im = 0; im < block_m; ++im) {
                for (int in = 0; in < block_n; ++in) {
                    sum[im * block_n + in] += tmp_x[im] * tmp_y[in];
                }
            }
        }
        __syncthreads();
    }

    int row = tile_row * tile_m + thread_row * block_m;
    int col = tile_col * tile_n + thread_col * block_n;

    for (int im = 0; im < block_m && row + im < m; ++im) {
        for (int in = 0; in < block_n && col + in < n; ++in) {
            Z[(row + im) * n + (col + in)] = sum[im * block_n + in];
        }
    }
}

template <typename T>
__global__ void cp_row(T* vec_row, const T* mat, unsigned i, unsigned n_col)
{
    auto vj     = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; vj < n_col; vj += stride) { vec_row[vj] = mat[i * n_col + vj]; }
}

template <typename T>
__global__ void cp_col(T* vec_col, const T* mat, unsigned j, unsigned n_row,
                       unsigned n_col)
{
    auto vi     = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; vi < n_row; vi += stride) { vec_col[vi] = mat[vi * n_col + j]; }
}

} // namespace MatrixOp