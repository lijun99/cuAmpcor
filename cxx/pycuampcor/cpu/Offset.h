/**
 * @file  Offset.h
 * @brief Peak locations of correlation surfaces and offsets (CPU backend)
 *
 * Implemented in Offset.cpp.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cpu {

/// the maximum of each surface of images, and its location (row, column)
void cuArraysMaxloc2D(ArraysView<const real_type> images, ArraysView<int2> maxloc,
    ArraysView<real_type> maxval, stream_t stream);

/// the same, within range pixels (rows, columns) starting at the pixel start of each surface
void cuArraysMaxloc2D(ArraysView<const real_type> images, const int2 start, const int2 range,
    ArraysView<int2> maxloc, ArraysView<real_type> maxval, stream_t stream);

/// the offsets (offsetInit - initOrigin)/initFactor + (offsetZoomIn - zoomInOrigin)/zoomInFactor
/// (one-pass workflow)
void cuSubPixelOffset(ArraysView<const int2> offsetInit, ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal, const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor, stream_t stream);

/// the offsets offsetInit - halfRangeInit + offsetZoomIn/(OverSampleRatioZoomin*OverSampleRatioRaw)
/// (two-pass workflow)
void cuSubPixelOffset2Pass(ArraysView<const int2> offsetInit, ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal, int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit, int yHalfRangeInit, stream_t stream);

/// the starting pixels of the secondary windows of the half search range (xNewRange, yNewRange)
/// around the peaks maxLoc, within the half search range (xOldRange, yOldRange) (in place), and
/// the shift of the peaks from their centers (when the peaks are near the edges)
void cuDetermineSecondaryExtractOffset(ArraysView<int2> maxLoc, ArraysView<int2> maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t stream);

} // namespace
//end of file
