#pragma once

#include "cuda_runtime.h"
#include <limits>

enum class BinaryOp { EQ, NE, GT, GE, LT, LE, MIN, MAX };

template <typename T>
struct ValueIndex {
    T   val;
    int idx;
} __attribute__((aligned(2 * std::max(sizeof(T), 4lu))));

namespace CommonOp
{
template <typename T>
__forceinline__ __device__ int eq(const T& lhs, const T& rhs)
{
    return lhs == rhs;
}

template <typename T>
__forceinline__ __device__ int ne(const T& lhs, const T& rhs)
{
    return lhs != rhs;
}

template <typename T>
__forceinline__ __device__ int gt(const T& lhs, const T& rhs)
{
    return lhs > rhs;
}

template <typename T>
__forceinline__ __device__ int ge(const T& lhs, const T& rhs)
{
    return lhs >= rhs;
}

template <typename T>
__forceinline__ __device__ int lt(const T& lhs, const T& rhs)
{
    return lhs < rhs;
}

template <typename T>
__forceinline__ __device__ int le(const T& lhs, const T& rhs)
{
    return lhs <= rhs;
}

template <typename T>
__forceinline__ __device__ T min(const T& lhs, const T& rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__forceinline__ __device__ T max(const T& lhs, const T& rhs)
{
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__forceinline__ __device__ void max_assign(volatile ValueIndex<T>*       dst,
                                           const volatile ValueIndex<T>* src)
{
    if (dst->val < src->val) {
        dst->val = src->val;
        dst->idx = src->idx;
    }
}

template <typename T>
__forceinline__ __device__ void max_assign(ValueIndex<T>&       dst,
                                           const ValueIndex<T>& src)
{
    if (dst.val < src.val) {
        dst.val = src.val;
        dst.idx = src.idx;
    }
}

template <typename T>
__forceinline__ __device__ void max_assign(ValueIndex<T>& dst, T val, int idx)
{
    if (dst.val < val) {
        dst.val = val;
        dst.idx = idx;
    }
}

template <typename T>
__forceinline__ __device__ void min_assign(volatile ValueIndex<T>*       dst,
                                           const volatile ValueIndex<T>* src)
{
    if (dst->val > src->val) {
        dst->val = src->val;
        dst->idx = src->idx;
    }
}

template <typename T>
__forceinline__ __device__ void min_assign(ValueIndex<T>&       dst,
                                           const ValueIndex<T>& src)
{
    if (dst.val > src.val) {
        dst.val = src.val;
        dst.idx = src.idx;
    }
}

template <typename T>
__forceinline__ __device__ void min_assign(ValueIndex<T>& dst, T val, int idx)
{
    if (dst.val > val) {
        dst.val = val;
        dst.idx = idx;
    }
}

template <typename T>
__forceinline__ __device__ T binary_op(BinaryOp op, const T& lhs, const T& rhs)
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

template <typename T, unsigned block_dim>
__forceinline__ __device__ void warp_sum(volatile T* smem, unsigned tid)
{
    if (block_dim >= 64) smem[tid] = smem[tid] + smem[tid + 32];
    if (block_dim >= 32) smem[tid] = smem[tid] + smem[tid + 16];
    if (block_dim >= 16) smem[tid] = smem[tid] + smem[tid + 8];
    if (block_dim >= 8) smem[tid] = smem[tid] + smem[tid + 4];
    if (block_dim >= 4) smem[tid] = smem[tid] + smem[tid + 2];
    if (block_dim >= 2) smem[tid] = smem[tid] + smem[tid + 1];
}

template <typename T, unsigned block_dim>
__forceinline__ __device__ void warp_max(volatile ValueIndex<T>* smem,
                                         unsigned                tid)
{
    if (block_dim >= 64) max_assign(smem + tid, smem + tid + 32);
    if (block_dim >= 32) max_assign(smem + tid, smem + tid + 16);
    if (block_dim >= 16) max_assign(smem + tid, smem + tid + 8);
    if (block_dim >= 8) max_assign(smem + tid, smem + tid + 4);
    if (block_dim >= 4) max_assign(smem + tid, smem + tid + 2);
    if (block_dim >= 2) max_assign(smem + tid, smem + tid + 1);
}

template <typename T, unsigned block_dim>
__forceinline__ __device__ void warp_min(volatile ValueIndex<T>* smem,
                                         unsigned                tid)
{
    if (block_dim >= 64) min_assign(smem + tid, smem + tid + 32);
    if (block_dim >= 32) min_assign(smem + tid, smem + tid + 16);
    if (block_dim >= 16) min_assign(smem + tid, smem + tid + 8);
    if (block_dim >= 8) min_assign(smem + tid, smem + tid + 4);
    if (block_dim >= 4) min_assign(smem + tid, smem + tid + 2);
    if (block_dim >= 2) min_assign(smem + tid, smem + tid + 1);
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

template <typename T, unsigned block_dim>
__global__ void sum_unroll(T* Z, const T* X, unsigned n)
{
    unsigned tid  = threadIdx.x;
    unsigned i    = blockIdx.x * (block_dim * 2) + threadIdx.x;
    unsigned grid = block_dim * 2 * gridDim.x;

    __shared__ T Zs[block_dim];
    Zs[tid] = (T) 0;
    __syncthreads();

    while (i < n) {
        Zs[tid] += X[i];
        if (i + block_dim < n) Zs[tid] += X[i + blockDim.x];
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

template <typename T, unsigned block_dim>
__global__ void max_unroll(ValueIndex<T>* Z, const T* X, unsigned n)
{
    unsigned tid  = threadIdx.x;
    unsigned i    = blockIdx.x * (block_dim * 2) + threadIdx.x;
    unsigned grid = block_dim * 2 * gridDim.x;

    __shared__ ValueIndex<T> Zs[block_dim];
    Zs[tid].val = std::numeric_limits<T>::lowest();
    Zs[tid].idx = -1;
    __syncthreads();

    while (i < n) {
        Zs[tid].val = X[i];
        Zs[tid].idx = i;
        if (i + block_dim < n) {
            max_assign(Zs[tid], X[i + blockDim.x], i + blockDim.x);
        }
        i += grid;
    }
    __syncthreads();

    if (block_dim >= 512) {
        if (tid < 256) { max_assign(Zs[tid], Zs[tid + 256]); }
        __syncthreads();
    }
    if (block_dim >= 256) {
        if (tid < 128) { max_assign(Zs[tid], Zs[tid + 128]); }
        __syncthreads();
    }
    if (block_dim >= 128) {
        if (tid < 64) { max_assign(Zs[tid], Zs[tid + 64]); }
        __syncthreads();
    }

    if (tid < 32) CommonOp::warp_max<T, block_dim>(Zs, tid);
    if (tid == 0) { Z[blockIdx.x] = Zs[0]; }
}

template <typename T, unsigned block_dim>
__global__ void min_unroll(ValueIndex<T>* Z, const T* X, unsigned n)
{
    unsigned tid  = threadIdx.x;
    unsigned i    = blockIdx.x * (block_dim * 2) + threadIdx.x;
    unsigned grid = block_dim * 2 * gridDim.x;

    __shared__ ValueIndex<T> Zs[block_dim];
    Zs[tid].val = std::numeric_limits<T>::max();
    Zs[tid].idx = -1;
    __syncthreads();

    while (i < n) {
        Zs[tid].val = X[i];
        Zs[tid].idx = i;
        if (i + block_dim < n) {
            min_assign(Zs[tid], X[i + blockDim.x], i + blockDim.x);
        }
        i += grid;
    }
    __syncthreads();

    if (block_dim >= 512) {
        if (tid < 256) { min_assign(Zs[tid], Zs[tid + 256]); }
        __syncthreads();
    }
    if (block_dim >= 256) {
        if (tid < 128) { min_assign(Zs[tid], Zs[tid + 128]); }
        __syncthreads();
    }
    if (block_dim >= 128) {
        if (tid < 64) { min_assign(Zs[tid], Zs[tid + 64]); }
        __syncthreads();
    }

    if (tid < 32) CommonOp::warp_min<T, block_dim>(Zs, tid);
    if (tid == 0) { Z[blockIdx.x] = Zs[0]; }
}

} // namespace CommonOp