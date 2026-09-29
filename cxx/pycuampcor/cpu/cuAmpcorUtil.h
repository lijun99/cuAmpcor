/*
 * @file cuAmpcorUtil.h
 * @brief Header file to include various routines for cuAmpcor (CPU backend)
 *
 *
 */

// code guard
#ifndef __CUAMPCORUTIL_H
#define __CUAMPCORUTIL_H

#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include "debug.h"
#include "cpuUtil.h"
#include "float2.h"

namespace pycuampcor::cpu {


//in cuArraysCopy.cpp: various utilities for copy images file in memory
void cuArraysCopyToBatch(cuArrays<image_complex_type> *image1, cuArrays<complex_type> *image2, int strideH, int strideW, stream_t stream);
void cuArraysCopyToBatchWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t stream);
void cuArraysCopyToBatchAbsWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t stream);
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<image_real_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t stream);
void cuArraysCopyC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, int strideH, int strideW, stream_t stream);

// same routine name overloaded for different data type
// extract data from a large image
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offset, stream_t);
template<typename T_in, typename T_out>
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset, stream_t);
void cuArraysCopyExtractAbs(cuArrays<complex_type> *imagesIn, cuArrays<real_type> *imagesOut, int2 offset, stream_t stream);

template<typename T>
void cuArraysCopyInsert(cuArrays<T> *in, cuArrays<T> *out, int offsetX, int offsetY, stream_t);

template<typename T_in, typename T_out>
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut,stream_t stream);
void cuArraysSetConstant(cuArrays<real_type> *imageIn, real_type value, stream_t stream);

void cuArraysR2C(cuArrays<real_type> *image1, cuArrays<complex_type> *image2, stream_t stream);
void cuArraysC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t stream);
void cuArraysAbs(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t stream);

// cuDeramp.cpp: deramping phase
void cuDeramp(int method, cuArrays<complex_type> *images, const int axis, stream_t stream);
void cuLinearDeramp(cuArrays<complex_type> *images, const int axis, stream_t stream);

// cuArraysPadding.cpp: various utilities for oversampling padding
void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, stream_t stream);

//in cuCorrNormalization.cpp: utilities to normalize the cross correlation function
void cuArraysSubtractMean(cuArrays<real_type> *images, stream_t stream);

// in cuCorrNormalizationSAT.cpp: to normalize the cross correlation function with sum area table
// sum area tables are computed in double precision for accuracy
void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
    cuArrays<double> *secondarySAT, cuArrays<double> *secondarySAT2, stream_t stream);


//in cuOffset.cpp: utilities for determining the max location of cross correlations or the offset
void cuArraysMaxloc2D(cuArrays<real_type> *images, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, stream_t stream);
void cuArraysMaxloc2D(cuArrays<real_type> *images, const int2 start, const int2 range, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, stream_t stream);
void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetRoomIn, cuArrays<real2_type> *offsetFinal, const int2 initOrigin, const int initFactor, const int2 zoomInOrigin, const int zoomInFactor, stream_t stream);
void cuSubPixelOffset2Pass(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn, cuArrays<real2_type> *offsetFinal, int OverSampleRatioZoomin, int OverSampleRatioRaw, int xHalfRangeInit,  int yHalfRangeInit, stream_t stream);
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift, int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t stream);

//in cuCorrTimeDomain.cpp: cross correlation in time domain
void cuCorrTimeDomain(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results, stream_t stream);

//in cuCorrFrequency.cpp: cross correlation in freq domain, also include fft correlatior class
void cuArraysElementMultiplyConjugate(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, real_type coef, stream_t stream);


// For SNR estimation on Correlation surface (Minyan Zhong)
// implemented in cuArraysCopy.cpp
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int2> *maxloc, stream_t stream);
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, stream_t stream);
// implemented in cuCorrNormalization.cpp
void cuArraysSumSquare(cuArrays<real_type> *images, cuArrays<real_type> *imagesSum, stream_t stream);
void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid, cuArrays<real_type> *imagesSum, cuArrays<int> *imagesValidCount, stream_t stream);


// implemented in cuEstimateStats.cpp
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, const int size, stream_t stream);
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<int> *corrValidCount, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, stream_t stream);

// implemented in cuEstimateStats.cpp
void cuEstimateVariance(cuArrays<real_type> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, const int templateSize, const int distance, cuArrays<real3_type> *covValue, stream_t stream);

} // namespace

#endif

// end of file
