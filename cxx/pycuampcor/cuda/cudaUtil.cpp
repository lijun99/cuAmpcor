#include "cudaUtil.h"
#include "backend.h"
#include "cuAmpcorParameter.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include "cudaError.h"

namespace pycuampcor::cuda {

int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        throw std::runtime_error("gpuDeviceInit() CUDA error: no devices supporting CUDA.");
    }

    if (devID < 0 || devID > device_count - 1) {
        throw std::runtime_error("gpuDeviceInit() Device " + std::to_string(devID) + " is not a valid GPU device.");
    }

    checkCudaErrors(cudaSetDevice(devID));
    printf("Using CUDA Device %d ...\n", devID);

    return devID;
}

void gpuDeviceList()
{
    int device_count = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    fprintf(stderr, "Detecting all CUDA devices ...\n");
    if (device_count == 0) {
        throw std::runtime_error("CUDA error: no devices supporting CUDA.");
    }

    for (int current_device = 0; current_device < device_count; ++current_device) {
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));

#if CUDART_VERSION < 13000   // computeMode field removed in CUDA 13
        if (deviceProp.computeMode == cudaComputeModeProhibited) {
            fprintf(stderr,
                    "CUDA Device [%d]: \"%s\" is not available: "
                    "device is running in <Compute Mode Prohibited>\n",
                    current_device, deviceProp.name);
            continue;
        }
#endif

        if (deviceProp.major < 1) {
            fprintf(stderr,
                    "CUDA Device [%d]: \"%s\" is not available: "
                    "device does not support CUDA\n",
                    current_device, deviceProp.name);
        } else {
            fprintf(stderr, "CUDA Device [%d]: \"%s\" is available.\n",
                    current_device, deviceProp.name);
        }
    }
}

int getSMCount(int devID)
{
    // use the currently active CUDA device if devID < 0
    if (devID < 0)
        checkCudaErrors(cudaGetDevice(&devID));

    // Retrieve device properties
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, devID));

    // Return the SM (Streaming Multiprocessor) count
    return prop.multiProcessorCount;
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
    checkCudaErrors(cudaMallocHost(&ptr, bytes));
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