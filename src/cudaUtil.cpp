#include "cudaUtil.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cudaError.h"

int gpuDeviceInit(int devID)
{
    int device_count;
    checkCudaErrors(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0 || devID > device_count - 1) {
        fprintf(stderr, "gpuDeviceInit() Device %d is not a valid GPU device. \n", devID);
        exit(EXIT_FAILURE);
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
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
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
    // Get the ID of the currently active CUDA device
    checkCudaErrors(cudaGetDevice(&devID));

    // Retrieve device properties
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, devID));

    // Return the SM (Streaming Multiprocessor) count
    return prop.multiProcessorCount;
}