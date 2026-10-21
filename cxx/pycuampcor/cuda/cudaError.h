/**
 * @file  cudaError.h
 * @brief Error checking of CUDA runtime and cuFFT calls, and of kernel launches
 *
 * checkCudaErrors(call): check the result of a CUDA runtime or cuFFT call
 * checkCudaAsyncErrors(): check the kernel launches (after them); in debug builds,
 *     also synchronize the device and check the kernel executions
 * Errors throw std::runtime_error (a python RuntimeError).
**/

#pragma once

#include "externals.h"
#include "debug.h"

namespace pycuampcor::cuda {

/// check the result of a CUDA runtime call: report an error with the failed call and its location
/// @param call,file,line the call (as text), and its location
void checkCudaErrorsImpl(cudaError_t result, const char *call, const char *file, int line);
/// check the result of a cuFFT call: report an error with the failed call and its location
/// @param call,file,line the call (as text), and its location
void checkCudaErrorsImpl(cufftResult result, const char *call, const char *file, int line);
/// check the kernel launches (and, with CUAMPCOR_DEBUG, their execution): report an error
/// with the location file:line of the check
void checkCudaAsyncErrorsImpl(const char *file, int line);

#define checkCudaErrors(call) \
    ::pycuampcor::cuda::checkCudaErrorsImpl((call), #call, __FILE__, __LINE__)
#define checkCudaAsyncErrors() \
    ::pycuampcor::cuda::checkCudaAsyncErrorsImpl(__FILE__, __LINE__)

} // namespace
