/**
 * @file  backend_externals.h
 * @brief The external headers and implementation choices of the CUDA backend (see externals.h)
 */

#pragma once

// cuda and its fft library
#include <cuda_runtime.h>
#include <cufft.h>

// the precision and the vector types (real_type, complex_type, float2, ...)
#include "data_types.h"

//end of file
