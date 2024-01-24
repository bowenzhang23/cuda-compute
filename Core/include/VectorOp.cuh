#pragma once

#include "cuda_runtime.h"

namespace VectorOp
{
template <typename T>
__global__ void axpbyc(T* Z, const T a, const T* X, const T b, const T* Y,
                       const T c, unsigned n)
{
    auto i      = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) { Z[i] = a * X[i] + b * Y[i] + c; }
}

template <typename T>
__global__ void cxamyb(T* Z, const T c, const T* X, const T a, const T* Y,
                       const T b, unsigned n)
{
    auto i      = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) { Z[i] = c * pow(X[i], a) * pow(Y[i], b); }
}
} // namespace VectorOp