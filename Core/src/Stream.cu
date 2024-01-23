#include "DeviceManager.cuh"
#include "Error.cuh"
#include "Stream.cuh"

Stream::Stream()
{
    CUDA_CHECK(cudaStreamCreate(&m_stream));
    Sync();
}

Stream::Stream(unsigned flags)
{
    CUDA_CHECK(cudaStreamCreateWithFlags(&m_stream, flags));
    Sync();
}

Stream::Stream(unsigned flags, int priority)
{
    CUDA_CHECK(cudaStreamCreateWithPriority(&m_stream, flags, priority));
    Sync();
}

Stream::~Stream()
{
    Sync();
    CUDA_CHECK(cudaStreamDestroy(m_stream));
}

void Stream::Sync() { CUDA_CHECK(cudaStreamSynchronize(m_stream)); }

Stream* Stream::Create() { return new Stream(); }

Stream* Stream::CreateNonBlocking()
{
    return new Stream(cudaStreamNonBlocking);
}

Stream* Stream::CreateNonBlocking(int p)
{
    const auto& device = DeviceManager::Instance().CurrentDevice();
    auto        pg     = device.StreamPriorityGreatest();
    auto        pl     = device.StreamPriorityLeast();
    p += pg;
    p = min(p, pl);

    fprintf(stdout, "Creating non-blocking stream with priority %d\n", p);
    return new Stream(cudaStreamNonBlocking, p);
}
