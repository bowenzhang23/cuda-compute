#include "Device.cuh"
#include "Error.cuh"
#include <sstream>

void Device::Init()
{
    CUDA_CHECK(cudaGetDeviceProperties(&m_prop, m_id));
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&m_sp_least, &m_sp_greatest));
}

std::string Device::ToString() const
{
    auto peak =
        2.0 * Prop().memoryClockRate * (Prop().memoryBusWidth / 8) / 1e6;
    std::ostringstream ss;
    ss << Prop().name << '\n';
    ss << "Memory Clock Rate (GHz): " << Prop().memoryClockRate / 1e6 << '\n';
    ss << "Memory Bus Width (bits): " << Prop().memoryBusWidth << '\n';
    ss << "Peak Memory Bandwidth (GB/s): " << peak << '\n';
    ss << "SM Count: " << Prop().multiProcessorCount << '\n';
    ss << "Max Threads per Thread Block: " << Prop().maxThreadsPerBlock << '\n';
    ss << "Max Threads per SM: " << Prop().maxThreadsPerMultiProcessor << '\n';
    ss << "Max Threads Blocks per SM: " << Prop().multiProcessorCount << '\n';
    ss << "StreamPriority: from " << StreamPriorityLeast() << " to "
       << StreamPriorityGreatest() << '\n';
    ss << "ComputeCapability: " << Prop().major << '.' << Prop().minor << '\n';

    return ss.str();
}
