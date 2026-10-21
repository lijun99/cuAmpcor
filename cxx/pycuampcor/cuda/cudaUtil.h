/**
 * @file cudaUtil.h
 * @brief Various cuda related parameters and utilities
 **/

#ifndef __CUDAUTIL_H
#define __CUDAUTIL_H

#include <cstddef>
#include <string>
#include <vector>

namespace pycuampcor::cuda {

// for 2D FFT
#define NRANK 2

//typical choices of number of threads in a block
// for processing 1D and 2D arrays
#define NTHREADS 512  //
#define NTHREADS2D 16 //

#define IDX2R(i,j,NJ) (((i)*(NJ))+(j))  //row-major order

#define IDIVUP(i,j) ((i+j-1)/j)

// compute the next integer in power of 2
inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}

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

#endif //__CUDAUTIL_H
//end of file
