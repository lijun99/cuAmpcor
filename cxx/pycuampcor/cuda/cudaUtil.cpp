#include "cudaUtil.h"
#include "backend.h"
#include "cuAmpcorParameter.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
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
    printf("Using CUDA Device %d ...\n", devID);
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

// backend hooks for the shared ampcor code

int backendInit(cuAmpcorParameter *param)
{
    return gpuDeviceInit(param->deviceID);
}

int backendNumWorkers(const cuAmpcorParameter *param)
{
    return param->nStreams;
}

void backendDefaultChunkSize(const cuAmpcorParameter *param, int &down, int &across)
{
    // windows in a chunk are processed in parallel; 2x the number of SMs keeps the GPU busy
    // (benchmarked on V100 and RTX PRO 6000 Blackwell), stacked down (SM/4) to share more rows
    const int sms = getSMCount(param->deviceID);
    down = std::max(sms / 4, 1);
    across = 8;
}

stream_t backendCreateStream()
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return stream;
}

void backendDestroyStream(stream_t stream)
{
    checkCudaErrors(cudaStreamDestroy(stream));
}

void backendSynchronize()
{
    checkCudaErrors(cudaDeviceSynchronize());
}

void backendCopyFromHost2D(void *dst, size_t dpitch, const void *src, size_t spitch,
    size_t widthInBytes, size_t height, stream_t stream)
{
    checkCudaErrors(cudaMemcpy2DAsync(dst, dpitch, src, spitch,
        widthInBytes, height, cudaMemcpyHostToDevice, stream));
}

void *backendAllocStaging(size_t bytes)
{
    void *ptr = nullptr;
    if (cudaMallocHost(&ptr, bytes) != cudaSuccess) {
        // e.g., limited page-locked memory; clear the error and use pageable copies
        cudaGetLastError();
        std::cerr << "Warning: failed to allocate " << bytes << " bytes of page-locked memory;"
                  << " using pageable memory to load the images" << std::endl;
        return nullptr;
    }
    return ptr;
}

void backendFreeStaging(void *ptr)
{
    if (ptr) checkCudaErrors(cudaFreeHost(ptr));
}

event_t backendCreateEvent()
{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void backendDestroyEvent(event_t event)
{
    checkCudaErrors(cudaEventDestroy(event));
}

void backendRecordEvent(event_t event, stream_t stream)
{
    checkCudaErrors(cudaEventRecord(event, stream));
}

void backendWaitEvent(event_t event)
{
    checkCudaErrors(cudaEventSynchronize(event));
}


} // namespace