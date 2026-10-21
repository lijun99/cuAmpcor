/**
 * @file  device.h
 * @brief CUDA devices: selection and properties, and the kernel launch settings
 **/

#pragma once

#include "externals.h"
#include "indexing.h"

namespace pycuampcor::cuda {

/// the rank of the FFTs (2D FFT)
constexpr int NRANK = 2;

/// a typical choice of the number of threads in a block, for processing 1D arrays
constexpr int NTHREADS = 512;
/// a typical choice of the number of threads in a block in each dimension, for processing 2D arrays
constexpr int NTHREADS2D = 16;

/// a CUDA device
struct DeviceInfo {
    int id;                    ///< the device id
    std::string name;          ///< the device name
    int smCount;               ///< number of streaming multiprocessors
    int major, minor;          ///< compute capability
    std::size_t totalMemory;   ///< global memory in bytes
};

/// select the device (devID) for the calling thread, and return its id (throws if not available)
int gpuDeviceInit(int devID=0);

/// list the available devices
std::vector<DeviceInfo> gpuDeviceList();

/// the number of streaming multiprocessors of the device devID (of the current device if devID < 0)
int getSMCount(int devID=0);

} // namespace
//end of file
