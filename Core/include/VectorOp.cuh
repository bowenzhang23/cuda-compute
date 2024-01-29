#pragma once

#include "cuda_runtime.h"

namespace VectorOp
{
template <typename T, unsigned block_dim>
__global__ void inner_unroll(T* Z, const T* X, const T* Y, unsigned n)
{
    unsigned tid  = threadIdx.x;
    unsigned i    = blockIdx.x * (block_dim * 2) + threadIdx.x;
    unsigned grid = block_dim * 2 * gridDim.x;
    if (i >= n) return;

    __shared__ T Zs[block_dim];
    Zs[tid] = (T) 0;

    while (i < n) {
        Zs[tid] += X[i] * Y[i];
        if (i + block_dim < n) Zs[tid] += X[i + blockDim.x] * Y[i + blockDim.x];
        i += grid;
    }
    __syncthreads();

    if (block_dim >= 512) {
        if (tid < 256) { Zs[tid] += Zs[tid + 256]; }
        __syncthreads();
    }
    if (block_dim >= 256) {
        if (tid < 128) { Zs[tid] += Zs[tid + 128]; }
        __syncthreads();
    }
    if (block_dim >= 128) {
        if (tid < 64) { Zs[tid] += Zs[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) CommonOp::warp_sum<T, block_dim>(Zs, tid);
    if (tid == 0) Z[blockIdx.x] = Zs[0];
}

template <typename T, unsigned tile_dim = 256, unsigned block_dim = 32>
__global__ void reversed(T* Z, const T* X, unsigned n)
{
    unsigned     i = threadIdx.x + blockIdx.x * tile_dim;
    __shared__ T Xs[tile_dim];
    for (int j = 0; j < tile_dim; j += block_dim) {
        if (i + j < n)
            Xs[threadIdx.x + j] = X[n - 1 - (i + j)];
        else
            Xs[threadIdx.x + j] = 0;
    }
    __syncthreads();

    for (int j = 0; j < tile_dim; j += block_dim) {
        if (i + j < n) Z[i + j] = Xs[threadIdx.x + j];
    }
}

} // namespace VectorOp