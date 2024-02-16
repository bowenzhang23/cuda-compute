#pragma once

#include "cuda_runtime.h"

namespace VectorOp
{
__host__ __device__ unsigned next_po2(unsigned n)
{
    unsigned n_bits = 0;
    unsigned n0     = n;
    while (n >>= 1) ++n_bits;
    return n0 == (unsigned) 1 << n_bits ? n0 : (unsigned) 1 << (n_bits + 1);
}

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

template <typename T>
__forceinline__ __device__ void exchange(T* a, T* b)
{
    T tmp = *a;
    *a    = *b;
    *b    = tmp;
}

template <typename T>
__global__ void sort(T* Z, unsigned k, unsigned j, unsigned n, bool ascending)
{
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (unsigned l = i ^ j; l > i) {
        if ((i & k) == 0 && (ascending ? Z[i] > Z[l] : Z[i] < Z[l]))
            exchange(Z + i, Z + l);
        if ((i & k) != 0 && (ascending ? Z[i] < Z[l] : Z[i] > Z[l]))
            exchange(Z + i, Z + l);
    }
}

template <typename T>
__global__ void sort_padding(T* Z, T* P, unsigned k, unsigned j, unsigned n,
                             bool ascending)
{
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (unsigned l = i ^ j; l > i) {
        if (l < n) {
            if ((i & k) == 0 && (ascending ? Z[i] > Z[l] : Z[i] < Z[l]))
                exchange(Z + i, Z + l);
            if ((i & k) != 0 && (ascending ? Z[i] < Z[l] : Z[i] > Z[l]))
                exchange(Z + i, Z + l);
        } else if (i < n) {
            if ((i & k) == 0 && (ascending ? Z[i] > P[l - n] : Z[i] < P[l - n]))
                exchange(Z + i, P + l - n);
            if ((i & k) != 0 && (ascending ? Z[i] < P[l - n] : Z[i] > P[l - n]))
                exchange(Z + i, P + l - n);
        } else {
            if ((i & k) == 0
                && (ascending ? P[i - n] > P[l - n] : P[i - n] < P[l - n]))
                exchange(P + i - n, P + l - n);
            if ((i & k) != 0
                && (ascending ? P[i - n] < P[l - n] : P[i - n] > P[l - n]))
                exchange(P + i - n, P + l - n);
        }
    }
}

} // namespace VectorOp