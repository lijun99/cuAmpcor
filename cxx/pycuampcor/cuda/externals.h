/**
 * @file  externals.h
 * @brief The external headers and implementation choices of the CUDA package
 */

#pragma once

// the external headers of the shared code
#include "../common/externals.h"

// cuda and its fft library
#include <cuda_runtime.h>
#include <cufft.h>

// the precision and the vector types (real_type, complex_type, float2, ...)
#include "data_types.h"

namespace pycuampcor::cuda {
    /// the stream on which the kernels and copies are queued
    using stream_t = cudaStream_t;
}

//end of file
