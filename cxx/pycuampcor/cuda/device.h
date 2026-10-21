/**
 * @file  device.h
 * @brief CUDA devices: selection and properties, and the kernel launch settings
 **/

#pragma once

#include "externals.h"
#include "indexing.h"

namespace pycuampcor::cuda {

// for 2D FFT
constexpr int NRANK = 2;

// typical choices of number of threads in a block
// for processing 1D and 2D arrays
constexpr int NTHREADS = 512;
constexpr int NTHREADS2D = 16;

// a CUDA device
struct DeviceInfo {
    int id;
    std::string name;
    int smCount;               ///< number of streaming multiprocessors
    int major, minor;          ///< compute capability
    std::size_t totalMemory;   ///< global memory in bytes
};

// select the device (devID) for the calling thread, and return its id
int gpuDeviceInit(int devID=0);

// list the available devices
std::vector<DeviceInfo> gpuDeviceList();

// Get the number of StreamMultiprocessor (of the current device if devID < 0)
int getSMCount(int devID=0);

} // namespace
//end of file
