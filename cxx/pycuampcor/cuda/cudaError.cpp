#include "cudaError.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <stdexcept>
#include <string>

namespace pycuampcor::cuda {

// report an error: throw an exception (translated to a python RuntimeError),
// or print the message if an exception is already propagating (e.g., in destructors during unwinding)
static void reportError(const std::string &message)
{
    if (std::uncaught_exceptions() > 0) {
        fprintf(stderr, "%s\n", message.c_str());
        return;
    }
    throw std::runtime_error(message);
}

static const char * errorString(cudaError_t result) { return cudaGetErrorString(result); }
static const char * errorString(cufftResult_t) { return "cufft error"; }

template<typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result) {
        char message[1024];
        snprintf(message, sizeof(message), "CUDA error at %s:%d code=%d(%s) \"%s\"",
                file, line, static_cast<unsigned int>(result), errorString(result), func);
        reportError(message);
    }
}

template void check(cudaError_t, char const *const, const char *const, int const);
template void check(cufftResult_t, char const *const, const char *const, int const);

void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        char message[1024];
        snprintf(message, sizeof(message), "%s(%i) : CUDA error : %s : (%d) %s.",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        reportError(message);
    }
}

} // namespace