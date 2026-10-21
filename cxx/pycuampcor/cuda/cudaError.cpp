#include "public.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

namespace pycuampcor::cuda {

namespace {

// report an error: throw an exception (translated to a python RuntimeError),
// or print the message if an exception is already propagating (e.g., in destructors during unwinding)
void reportError(const std::string &message)
{
    if (std::uncaught_exceptions() > 0) {
        logWarning(message);
        return;
    }
    throw std::runtime_error(message);
}

const char *cufftErrorName(cufftResult result)
{
    switch (result) {
        case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
        case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
        default: return "unknown cuFFT error";
    }
}

std::string location(const char *file, int line)
{
    return std::string(file) + ":" + std::to_string(line);
}

} // namespace

void checkCudaErrorsImpl(cudaError_t result, const char *call, const char *file, int line)
{
    if (result != cudaSuccess) {
        // clear the (non-sticky) error, so that it is not reported again by later checks
        cudaGetLastError();
        reportError("CUDA error " + std::string(cudaGetErrorName(result)) + " ("
            + cudaGetErrorString(result) + ") in " + call + " at " + location(file, line));
    }
}

void checkCudaErrorsImpl(cufftResult result, const char *call, const char *file, int line)
{
    if (result != CUFFT_SUCCESS)
        reportError("cuFFT error " + std::to_string(static_cast<int>(result)) + " ("
            + cufftErrorName(result) + ") in " + call + " at " + location(file, line));
}

void checkCudaAsyncErrorsImpl(const char *file, int line)
{
    // launch errors (clearing the non-sticky ones)
    checkCudaErrorsImpl(cudaGetLastError(), "kernel launch", file, line);
#ifdef CUAMPCOR_DEBUG
    // execution errors, reported at the failing kernel
    checkCudaErrorsImpl(cudaDeviceSynchronize(), "kernel execution", file, line);
#endif
}

} // namespace
