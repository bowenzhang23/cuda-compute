#pragma once

#include "cuda_runtime.h"
#include <string_view>

/**
 * @brief Query device properties
 *
 * https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
 * https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
 */
class Device
{
public:
    explicit Device(int id) : m_id(id) { Init(); }
    inline int ID() const { return m_id; }
    inline int StreamPriorityLeast() const { return m_sp_least; }
    inline int StreamPriorityGreatest() const { return m_sp_greatest; }
    inline const cudaDeviceProp& Prop() const { return m_prop; }
    std::string                  ToString() const;

private:
    void Init();

private:
    int            m_id;
    int            m_sp_least;
    int            m_sp_greatest;
    cudaDeviceProp m_prop;
};
