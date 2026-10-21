/**
 * @file  ArraysCopy.h
 * @brief Copy, extract, insert and convert batches of images (CPU backend)
 *
 * Implemented in ArraysCopy.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysCopyToBatch(array_type<image_complex_type> *image1, array_type<complex_type> *image2,
    int strideH, int strideW, stream_t stream);

void cuArraysCopyToBatchWithOffset(array_type<image_complex_type> *image1, const int inNX,
    const int inNY, array_type<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyToBatchAbsWithOffset(array_type<image_complex_type> *image1, const int inNX,
    const int inNY, array_type<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyToBatchWithOffsetR2C(array_type<image_real_type> *image1, const int inNX,
    const int inNY, array_type<complex_type> *image2, const int *offsetH, const int* offsetW,
    stream_t stream);

void cuArraysCopyC2R(array_type<complex_type> *image1, array_type<real_type> *image2, int strideH,
    int strideW, stream_t stream);

template<typename T>
void cuArraysCopyExtract(array_type<T> *imagesIn, array_type<T> *imagesOut, array_type<int2> *offset,
    stream_t);

template<typename T_in, typename T_out>
void cuArraysCopyExtract(array_type<T_in> *imagesIn, array_type<T_out> *imagesOut, int2 offset,
    stream_t);

void cuArraysCopyExtractAbs(array_type<complex_type> *imagesIn, array_type<real_type> *imagesOut,
    int2 offset, stream_t stream);

template<typename T>
void cuArraysCopyInsert(array_type<T> *in, array_type<T> *out, int offsetX, int offsetY, stream_t);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(array_type<T_in> *imageIn, array_type<T_out> *imageOut,stream_t stream);

void cuArraysSetConstant(array_type<real_type> *imageIn, real_type value, stream_t stream);

void cuArraysR2C(array_type<real_type> *image1, array_type<complex_type> *image2, stream_t stream);

void cuArraysC2R(array_type<complex_type> *image1, array_type<real_type> *image2, stream_t stream);

void cuArraysAbs(array_type<complex_type> *image1, array_type<real_type> *image2, stream_t stream);

void cuArraysCopyExtractCorr(array_type<real_type> *imagesIn, array_type<real_type> *imagesOut,
    array_type<int2> *maxloc, stream_t stream);

void cuArraysCopyExtractCorr(array_type<real_type> *imagesIn, array_type<real_type> *imagesOut,
    array_type<int> *imagesValid, array_type<int2> *maxloc, stream_t stream);

void cuArraysCopyExtractCorr(array_type<real_type> *imagesIn, array_type<real_type> *imagesOut,
    array_type<int> *imagesValid, array_type<int2> *maxloc, int stride, int2 validStart,
    int2 validRange, stream_t stream);

} // namespace
//end of file
