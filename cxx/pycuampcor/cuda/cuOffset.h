/**
 * @file  cuOffset.h
 * @brief Peak locations of correlation surfaces and offsets (CUDA backend)
 *
 * Implemented in cuOffset.cu.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cuda {

void cuArraysMaxloc2D(cuArrays<real_type> *images, cuArrays<int2> *maxloc,
    cuArrays<real_type> *maxval, stream_t stream);

void cuArraysMaxloc2D(cuArrays<real_type> *images, const int2 start, const int2 range,
    cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, stream_t stream);

void cuSubPixelOffset(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetRoomIn,
    cuArrays<real2_type> *offsetFinal, const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor, stream_t stream);

void cuSubPixelOffset2Pass(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn,
    cuArrays<real2_type> *offsetFinal, int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit, int yHalfRangeInit, stream_t stream);

void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t stream);

} // namespace
//end of file
