/**
 * @file  cuArraysPadding.h
 * @brief Zero padding of spectra for the FFT oversampling (CPU backend)
 *
 * Implemented in cuArraysPadding.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2,
    stream_t stream);

} // namespace
//end of file
