/**
 * @file  backend.cpp
 * @brief Backend hooks for the shared ampcor code, CUDA version (see backend.h)
 */

#include "backend.h"
#include "device.h"
#include "cudaError.h"
#include "cuAmpcorParameter.h"
#include "logging.h"

namespace pycuampcor::cuda {

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
        logWarning("failed to allocate ", bytes, " bytes of page-locked memory;",
                   " using pageable memory to load the images");
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