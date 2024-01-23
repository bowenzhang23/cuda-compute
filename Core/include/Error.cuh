#pragma once

#include "cuda_runtime.h"
#include <cstdio>

#define CUDA_ERRORCODE_CHECK_ENABLED
#define CUDA_ERRORCODE_STRICTCHECK_ENABLED

#define CUDA_CHECK(errno) __cuda_check(errno, __FILE__, __LINE__)
#define CUDA_CHECK_LAST() __cuda_check_last(__FILE__, __LINE__)

inline void __cuda_check(cudaError_t errno, const char* file, const int line)
{
#ifdef CUDA_ERRORCODE_CHECK_ENABLED
    if (errno != cudaSuccess) {
        fprintf(stderr, "CUDA_CHECK() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(errno));
        exit(-1);
    }
#ifdef CUDA_ERRORCODE_STRICTCHECK_ENABLED
    errno = cudaDeviceSynchronize();
    if (errno != cudaSuccess) {
        fprintf(stderr,
                "cudaDeviceSynchronize() after CUDA_CHECK() failed at "
                "%s:%i : %s\n",
                file, line, cudaGetErrorString(errno));
        exit(-1);
    }
#endif
#endif
}

inline void __cuda_check_last(const char* file, const int line)
{
#ifdef CUDA_ERRORCODE_CHECK_ENABLED
    cudaError_t errno = cudaGetLastError();
    if (errno != cudaSuccess) {
        fprintf(stderr, "CUDA_CHECK_LAST() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(errno));
        exit(-1);
    }

#ifdef CUDA_ERRORCODE_STRICTCHECK_ENABLED
    errno = cudaDeviceSynchronize();
    if (errno != cudaSuccess) {
        fprintf(stderr,
                "cudaDeviceSynchronize() after CUDA_CHECK_LAST() failed at "
                "%s:%i : %s\n",
                file, line, cudaGetErrorString(errno));
        exit(-1);
    }
#endif
#endif
}
