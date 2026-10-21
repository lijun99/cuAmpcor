/**
 * @file  Offset.h
 * @brief Peak locations of correlation surfaces and offsets (CPU backend)
 *
 * Implemented in Offset.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysMaxloc2D(array_type<real_type> *images, array_type<int2> *maxloc,
    array_type<real_type> *maxval, stream_t stream);

void cuArraysMaxloc2D(array_type<real_type> *images, const int2 start, const int2 range,
    array_type<int2> *maxloc, array_type<real_type> *maxval, stream_t stream);

void cuSubPixelOffset(array_type<int2> *offsetInit, array_type<int2> *offsetRoomIn,
    array_type<real2_type> *offsetFinal, const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor, stream_t stream);

void cuSubPixelOffset2Pass(array_type<int2> *offsetInit, array_type<int2> *offsetZoomIn,
    array_type<real2_type> *offsetFinal, int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit, int yHalfRangeInit, stream_t stream);

void cuDetermineSecondaryExtractOffset(array_type<int2> *maxLoc, array_type<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t stream);

} // namespace
//end of file
