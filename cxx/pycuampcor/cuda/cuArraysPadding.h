/**
 * @file  cuArraysPadding.h
 * @brief Zero padding of spectra for the FFT oversampling (CUDA backend)
 *
 * Implemented in cuArraysPadding.cu.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cuda {

void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2,
    stream_t stream);

} // namespace
//end of file
