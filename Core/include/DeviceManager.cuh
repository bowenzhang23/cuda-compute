#pragma once

#include "Device.cuh"
#include <vector>

class DeviceManager
{
public:
    static DeviceManager& Instance();
    static const Device&  Curr() { return Instance().CurrentDevice(); }
    void                  UseDevice(int id) const;
    const Device&         CurrentDevice() const;
    inline const Device&  At(int id) const { return m_devices.at(id); }
    inline int            Num() const { return m_devices.size(); }
    std::string           ToString() const;

private:
    DeviceManager();

private:
    std::vector<Device> m_devices;
};