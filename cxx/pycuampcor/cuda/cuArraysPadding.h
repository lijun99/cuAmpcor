/**
 * @file  cuArraysPadding.h
 * @brief Zero padding of spectra for the FFT oversampling (CUDA backend)
 *
 * Implemented in cuArraysPadding.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// pad the spectra image1 with zeros in the middle to the larger image2 (for the FFT oversampling),
/// scaled by 1/(height*width) of image1 (the inverse FFT normalization)
void cuArraysFFTPaddingMany(ArraysView<const complex_type> image1, ArraysView<complex_type> image2,
    stream_t stream);

} // namespace
//end of file
