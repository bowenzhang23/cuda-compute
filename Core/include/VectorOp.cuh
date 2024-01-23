#pragma once

#include "cuda_runtime.h"

namespace VectorOp
{
template <typename T>
__global__ void axpby(T* Z, const T a, const T* X, const T b, const T* Y,
                      unsigned n)
{
    auto i      = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) { Z[i] = a * X[i] + b * Y[i]; }
}
} // namespace VectorOp