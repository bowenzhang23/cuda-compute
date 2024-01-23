
#include "Error.cuh"
#include "Timer.cuh"

void Timer::Tick(const cudaStream_t stream)
{
    m_isRecording = !m_isRecording;
    if (m_isRecording) // start
    {
        m_elapsedTime = 0.f;
        CUDA_CHECK(cudaEventRecord(m_begin, stream));
    } else // end
    {
        CUDA_CHECK(cudaEventRecord(m_end, stream));
        CUDA_CHECK(cudaEventSynchronize(m_begin));
        CUDA_CHECK(cudaEventSynchronize(m_end));
        CUDA_CHECK(cudaEventElapsedTime(&m_elapsedTime, m_begin, m_end));
    }
}

Timer::Timer()
    : m_isRecording(false), m_elapsedTime(0.f), m_begin(), m_end()
{
    CUDA_CHECK(cudaEventCreate(&m_begin));
    CUDA_CHECK(cudaEventCreate(&m_end));
}