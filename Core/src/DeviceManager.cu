#include "DeviceManager.cuh"
#include "Error.cuh"
#include <sstream>

DeviceManager& DeviceManager::Instance()
{
    static DeviceManager dm;
    return dm;
}

void DeviceManager::UseDevice(int id) const
{
    if (id < (int) m_devices.size()) CUDA_CHECK(cudaSetDevice(id));
}

const Device& DeviceManager::CurrentDevice() const
{
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return m_devices.at(id);
}

std::string DeviceManager::ToString() const
{
    std::ostringstream ss;
    for (int i = 0; i < Num(); ++i) {
        ss << "Device Number: " << i << '\n';
        ss << At(i).ToString();
    }

    return ss.str();
}

DeviceManager::DeviceManager()
{
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) { m_devices.emplace_back(i); }
}