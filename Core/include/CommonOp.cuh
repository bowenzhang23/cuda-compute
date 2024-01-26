#pragma once

#include "cuda_runtime.h"

enum class BinaryOp { EQ, NE, GT, GE, LT, LE, MIN, MAX };

namespace CommonOp
{
template <typename T>
__device__ int eq(const T& lhs, const T& rhs)
{
    return lhs == rhs;
}

template <typename T>
__device__ int ne(const T& lhs, const T& rhs)
{
    return lhs != rhs;
}

template <typename T>
__device__ int gt(const T& lhs, const T& rhs)
{
    return lhs > rhs;
}

template <typename T>
__device__ int ge(const T& lhs, const T& rhs)
{
    return lhs >= rhs;
}

template <typename T>
__device__ int lt(const T& lhs, const T& rhs)
{
    return lhs < rhs;
}

template <typename T>
__device__ int le(const T& lhs, const T& rhs)
{
    return lhs <= rhs;
}

template <typename T>
__device__ T min(const T& lhs, const T& rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__device__ T max(const T& lhs, const T& rhs)
{
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__device__ T binary_op(BinaryOp op, const T& lhs, const T& rhs)
{
    switch (op) {
        case BinaryOp::EQ: return eq(lhs, rhs);
        case BinaryOp::NE: return ne(lhs, rhs);
        case BinaryOp::GT: return gt(lhs, rhs);
        case BinaryOp::GE: return ge(lhs, rhs);
        case BinaryOp::LT: return lt(lhs, rhs);
        case BinaryOp::LE: return le(lhs, rhs);
        case BinaryOp::MIN: return min(lhs, rhs);
        case BinaryOp::MAX: return max(lhs, rhs);
        default: return 0;
    }
}

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

template <typename T1, typename T2>
__global__ void binary(T1* Z, const T2* X, const T2* Y, BinaryOp op, unsigned n)
{
    auto i      = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) { Z[i] = binary_op(op, X[i], Y[i]); }
}

template <typename T1, typename T2>
__global__ void binary(T1* Z, const T2* X, T2 y, BinaryOp op, unsigned n)
{
    auto i      = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) { Z[i] = binary_op(op, X[i], y); }
}

} // namespace CommonOp