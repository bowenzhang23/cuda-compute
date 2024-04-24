#pragma once

#include "cuda_runtime.h"

class Stream
{
public:
    Stream();
    explicit Stream(unsigned flags);
    Stream(unsigned flags, int priority);
    ~Stream();
    Stream(Stream&)  = delete;
    Stream(Stream&&) = delete;

    void                       Sync();
    inline size_t              ID() const { return (size_t) m_stream; }
    inline const cudaStream_t& Cuda() const { return m_stream; }

public:
    static Stream* Create();
    static Stream* CreateDefault();
    static Stream* CreateNonBlocking();
    static Stream* CreateNonBlocking(int p /* relative priority */);

private:
    cudaStream_t m_stream;
};