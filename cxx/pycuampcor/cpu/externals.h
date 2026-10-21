/**
 * @file  externals.h
 * @brief The external headers and implementation choices of the CPU package
 */

#pragma once

// the external headers of the shared code
#include "../common/externals.h"

// the fft library
#include <fftw3.h>

// the precision and the vector types (real_type, complex_type, float2, ...)
#include "data_types.h"

namespace pycuampcor::cpu {
    /// the stream argument of the kernels, for the same signatures as the CUDA kernels;
    /// the CPU kernels run synchronously in the calling thread, so it carries nothing
    struct stream_t {};
}

//end of file
