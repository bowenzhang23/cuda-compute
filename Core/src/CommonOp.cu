#include "CommonOp.cuh"

namespace CommonOp
{

template <>
__forceinline__ __device__ float atomicMax(float* address, float val)
{
    unsigned* p        = (unsigned*) address;
    float     flt_val  = *address;
    unsigned  expected = __float2uint_rn(flt_val);
    unsigned  desired  = __float2uint_rn(max(flt_val, val));
    unsigned  old      = atomicCAS(p, expected, desired);
    while (old != expected) {
        expected = old;
        desired  = __float2uint_rn(max(__uint2float_rn(old), val));
        old      = atomicCAS(p, expected, desired);
    }
    return __uint2float_rn(old);
}

template <>
__forceinline__ __device__ float atomicMin(float* address, float val)
{
    unsigned* p        = (unsigned*) address;
    float     flt_val  = *address;
    unsigned  expected = __float2uint_rn(flt_val);
    unsigned  desired  = __float2uint_rn(min(flt_val, val));
    unsigned  old      = atomicCAS(p, expected, desired);
    while (old != expected) {
        expected = old;
        desired  = __float2uint_rn(min(__uint2float_rn(old), val));
        old      = atomicCAS(p, expected, desired);
    }
    return __uint2float_rn(old);
}

} // namespace CommonOp