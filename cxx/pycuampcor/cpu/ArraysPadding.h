/**
 * @file  ArraysPadding.h
 * @brief Zero padding of spectra for the FFT oversampling (CPU backend)
 *
 * Implemented in ArraysPadding.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysFFTPaddingMany(array_type<complex_type> *image1, array_type<complex_type> *image2,
    stream_t stream);

} // namespace
//end of file
