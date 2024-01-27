#pragma once

#include "cuda_runtime.h"

namespace VectorOp
{
template <typename T, unsigned nsmem>
__global__ void inner(T* Z, const T* X, const T* Y, unsigned n)
{
    unsigned tid = threadIdx.x;
    unsigned i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    if (i >= n) return;

    __shared__ T Zs[nsmem];

    Zs[tid] = X[i] * Y[i];
    if (i + blockDim.x < n) {
        Zs[tid] += X[i + blockDim.x] * Y[i + blockDim.x];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { Zs[tid] += Zs[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) Z[blockIdx.x] = Zs[0];
}
} // namespace VectorOp