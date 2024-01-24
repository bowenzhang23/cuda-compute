#pragma once

#include "Error.cuh"
#include "Timer.cuh"
#include "cuda_runtime.h"
#include <vector>

// #define DEBUG_CONSTRUCTOR
#define DEBUG_PERFORMANCE

template <typename T>
class CudaData
{
public:
    CudaData(size_t size, cudaStream_t stream = 0);
    CudaData(const T* hmem, size_t size, cudaStream_t stream = 0);
    CudaData(const CudaData& other);
    CudaData(CudaData&& other);
    CudaData& operator=(const CudaData& other);
    CudaData& operator=(CudaData&& other);

    virtual ~CudaData();

    virtual std::vector<T>             ToCPU() const;
    virtual inline std::vector<size_t> Shape() const { return {}; }

    inline T*                  Data() { return m_dmem; }
    inline const T*            Data() const { return m_dmem; }
    inline const cudaStream_t& S() const { return m_stream; }

protected:
    cudaStream_t m_stream;
    size_t       m_size;
    T*           m_dmem = nullptr;
    T*           m_hmem = nullptr;
};

template <typename T>
inline CudaData<T>::CudaData(size_t size, cudaStream_t stream)
    : m_size(size), m_stream(stream)
{
    m_hmem = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    CUDA_CHECK(cudaMemsetAsync((void*) m_dmem, 0, m_size, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

template <typename T>
inline CudaData<T>::CudaData(const T* hmem, size_t size, cudaStream_t stream)
    : m_size(size), m_stream(stream)
{
    m_hmem = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, m_hmem, m_size, cudaMemcpyHostToDevice,
                               m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

template <typename T>
inline CudaData<T>::CudaData(const CudaData& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData copy ctor\n");
#endif
    if (this == &other) return;
    m_size   = other.m_size;
    m_stream = other.m_stream;
    m_hmem   = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, other.m_hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, other.m_dmem, m_size,
                               cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

template <typename T>
inline CudaData<T>::CudaData(CudaData&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData move ctor\n");
#endif
    if (this == &other) return;
    m_size   = other.m_size;
    m_stream = other.m_stream;
    std::swap(m_hmem, other.m_hmem);
    std::swap(m_dmem, other.m_dmem);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

template <typename T>
inline CudaData<T>& CudaData<T>::operator=(const CudaData& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData copy assignment\n");
#endif
    if (this == &other) return *this;
    m_size   = other.m_size;
    m_stream = other.m_stream;
    m_hmem   = (T*) calloc(1, m_size);
    CUDA_CHECK(cudaMallocAsync((void**) &m_dmem, m_size, m_stream));
    memcpy(m_hmem, other.m_hmem, m_size);
    CUDA_CHECK(cudaMemcpyAsync(m_dmem, other.m_dmem, m_size,
                               cudaMemcpyDeviceToDevice, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    return *this;
}

template <typename T>
inline CudaData<T>& CudaData<T>::operator=(CudaData&& other)
{
#ifdef DEBUG_CONSTRUCTOR
    fprintf(stdout, "CudaData move assignment\n");
#endif
    if (this == &other) return *this;
    m_size   = other.m_size;
    m_stream = other.m_stream;
    std::swap(m_hmem, other.m_hmem);
    std::swap(m_dmem, other.m_dmem);
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    return *this;
}

template <typename T>
inline CudaData<T>::~CudaData()
{
    free(m_hmem);
    CUDA_CHECK(cudaFreeAsync((void*) m_dmem, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

template <typename T>
inline std::vector<T> CudaData<T>::ToCPU() const
{
    auto           count = m_size / sizeof(T);
    std::vector<T> hmem(count);
    CUDA_CHECK(cudaMemcpyAsync(m_hmem, m_dmem, m_size, cudaMemcpyDeviceToHost,
                               m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    memcpy((void*) hmem.data(), m_hmem, m_size);
    return hmem;
}
