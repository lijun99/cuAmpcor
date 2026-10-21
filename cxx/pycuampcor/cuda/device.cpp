/**
 * @file  device.cpp
 * @brief CUDA devices: selection and properties
 */

#include "device.h"
#include "logging.h"
#include "cudaError.h"

namespace pycuampcor::cuda {

int gpuDeviceInit(int devID)
{
    int count = 0;
    checkCudaErrors(cudaGetDeviceCount(&count));
    if (devID < 0 || devID >= count)
        throw std::runtime_error("Invalid CUDA device " + std::to_string(devID)
            + " (" + std::to_string(count) + " devices available)");

    checkCudaErrors(cudaSetDevice(devID));
    logInfo("Using CUDA Device ", devID, " ...");
    return devID;
}

std::vector<DeviceInfo> gpuDeviceList()
{
    int count = 0;
    checkCudaErrors(cudaGetDeviceCount(&count));

    std::vector<DeviceInfo> devices;
    for (int id = 0; id < count; id++) {
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, id));
        devices.push_back({id, prop.name, prop.multiProcessorCount,
                           prop.major, prop.minor, prop.totalGlobalMem});
    }
    return devices;
}

int getSMCount(int devID)
{
    // use the currently active CUDA device if devID < 0
    if (devID < 0)
        checkCudaErrors(cudaGetDevice(&devID));

    int count;
    checkCudaErrors(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, devID));
    return count;
}

} // namespace
