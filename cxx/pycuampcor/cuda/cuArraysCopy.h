/**
 * @file  cuArraysCopy.h
 * @brief Copy, extract, insert and convert batches of images (CUDA backend)
 *
 * Implemented in cuArraysCopy.cu.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cuda {

void cuArraysCopyToBatch(cuArrays<image_complex_type> *image1, cuArrays<complex_type> *image2,
    int strideH, int strideW, stream_t stream);

void cuArraysCopyToBatchWithOffset(cuArrays<image_complex_type> *image1, const int inNX,
    const int inNY, cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyToBatchAbsWithOffset(cuArrays<image_complex_type> *image1, const int inNX,
    const int inNY, cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyToBatchWithOffsetR2C(cuArrays<image_real_type> *image1, const int inNX,
    const int inNY, cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, int strideH,
    int strideW, stream_t stream);

template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offset,
    stream_t);

template<typename T_in, typename T_out>
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset,
    stream_t);

void cuArraysCopyExtractAbs(cuArrays<complex_type> *imagesIn, cuArrays<real_type> *imagesOut,
    int2 offset, stream_t stream);

template<typename T>
void cuArraysCopyInsert(cuArrays<T> *in, cuArrays<T> *out, int offsetX, int offsetY, stream_t);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut,stream_t stream);

void cuArraysSetConstant(cuArrays<real_type> *imageIn, real_type value, stream_t stream);

void cuArraysR2C(cuArrays<real_type> *image1, cuArrays<complex_type> *image2, stream_t stream);

void cuArraysC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t stream);

void cuArraysAbs(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t stream);

void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    cuArrays<int2> *maxloc, stream_t stream);

void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, stream_t stream);

void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, int stride, int2 validStart,
    int2 validRange, stream_t stream);

} // namespace
//end of file
