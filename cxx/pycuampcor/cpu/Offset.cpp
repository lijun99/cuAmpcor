/*
 * @file Offset.cpp
 * @brief Utilities used to determine the offset field (CPU backend)
 *
 */

// my module dependencies
#include "Offset.h"
#include "float2.h"

namespace pycuampcor::cpu {

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[in] start starting search pixel
 * @param[in] range search range
 * @param[out] maxloc arrays to hold the max locations
 * @param[out] maxval arrays to hold the max values
 * @note This routine is overloaded with the routine without start/range
 */
void cuArraysMaxloc2D(ArraysView<const real_type> images,
                      const int2 start, const int2 range,
                      ArraysView<int2> maxloc, ArraysView<real_type> maxval, stream_t)
{
    for(int imageIdx = 0; imageIdx < images.count(); imageIdx++) {
        const ArraysView<const real_type> image = images.array(imageIdx);
        real_type val = -REAL_MAX;
        int2 loc = start;
        for(int idx = start.x; idx < start.x + range.x; idx++) {
            const real_type *row = &image(idx, start.y);
            // the row maximum (if larger than the current maximum), vectorized
            real_type rowMax = val;
            #pragma omp simd reduction(max:rowMax)
            for(int idy = 0; idy < range.y; idy++)
                rowMax = (rowMax < row[idy]) ? row[idy] : rowMax;
            // only rows with a larger value are searched for the (first) location
            if(val < rowMax)
                for(int idy = 0; idy < range.y; idy++)
                    if(val < row[idy]) {
                        val = row[idy];
                        loc = make_int2(idx, start.y + idy);
                    }
        }
        maxloc[imageIdx] = loc;
        maxval[imageIdx] = val;
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxloc arrays to hold the max locations
 * @param[out] maxval arrays to hold the max values
 */
void cuArraysMaxloc2D(ArraysView<const real_type> images,
                      ArraysView<int2> maxloc, ArraysView<real_type> maxval, stream_t stream)
{
    // if no start and range are provided, use the whole image
    cuArraysMaxloc2D(images, make_int2(0, 0), make_int2(images.height(), images.width()),
        maxloc, maxval, stream);
}

/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] initOrigin, initFactor origin and scale factor of offsetInit
 * @param[in] zoomInOrigin, zoomInFactor origin and scale factor of offsetZoomIn
 */
void cuSubPixelOffset(ArraysView<const int2> offsetInit,
    ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal,
    const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor,
    stream_t)
{
    const int size = offsetInit.size();
    const real_type initRatio = 1.0f/static_cast<real_type>(initFactor);
    const real_type zoomInRatio = 1.0f/static_cast<real_type>(zoomInFactor);
    for(int idx = 0; idx < size; idx++) {
        const int2 init = offsetInit[idx];
        const int2 zoomIn = offsetZoomIn[idx];
        offsetFinal[idx].x = initRatio*(init.x-initOrigin.x) + zoomInRatio*(zoomIn.x - zoomInOrigin.x);
        offsetFinal[idx].y = initRatio*(init.y-initOrigin.y) + zoomInRatio*(zoomIn.y - zoomInOrigin.y);
    }
}

/**
 * Determine the final offset value for the two-pass workflow
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] OversampleRatioZoomIn the correlation surface oversampling factor
 * @param[in] OversampleRatioRaw the oversampling factor of reference/secondary windows before cross-correlation
 * @param[in] xHalfRangInit the original half search range along x, to be subtracted
 * @param[in] yHalfRangInit the original half search range along y, to be subtracted
 *
 * Final offset =  pixel size offset (offsetInit - half search range)
 *     + subpixel offset (offsetZoomIn / overall oversampling factor)
 */
void cuSubPixelOffset2Pass(ArraysView<const int2> offsetInit, ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit,
    stream_t)
{
    const int size = offsetInit.size();
    const real_type OSratio = real_type(1)/(OverSampleRatioZoomin*OverSampleRatioRaw);
    const real_type xoffset = xHalfRangeInit ;
    const real_type yoffset = yHalfRangeInit ;
    for(int idx = 0; idx < size; idx++) {
        offsetFinal[idx].x = OSratio*(offsetZoomIn[idx].x) + offsetInit[idx].x - xoffset;
        offsetFinal[idx].y = OSratio*(offsetZoomIn[idx].y) + offsetInit[idx].y - yoffset;
    }
}

// compute the start of extraction and the shift of center, see the cuda version for details
static inline int2 adjustOffset(const int oldRange, const int newRange, const int maxloc)
{
    // shift the max location by -newRange to find the start
    int start = maxloc - newRange;
    // if start is within the range, the max location will be in the center
    int shift = 0;
    // right boundary
    int rbound = 2*(oldRange-newRange);
    if(start<0)     // if exceeding the limit on the left
    {
        // set start at 0 and record the shift of center
        // (negative, the max location is on the left of the extracted center)
        shift = start;
        start = 0;
    }
    else if(start > rbound ) // if exceeding the limit on the right
    {
        shift = start-rbound;
        start = rbound;
    }
    return make_int2(start, shift);
}

/**
 * Determine the secondary window extract offset from the max location
 * @param[in,out] maxLoc the max locations as input, and the extraction starting pixels as output
 * @param[out] maxLocShift the shift of the max location from the extraction center
 * @param[in] xOldRange, yOldRange are (half) search ranges in first step
 * @param[in] xNewRange, yNewRange are (half) search range
 */
void cuDetermineSecondaryExtractOffset(ArraysView<int2> maxLoc, ArraysView<int2> maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t)
{
    const int size = maxLoc.size();
    for(int imageIndex = 0; imageIndex < size; imageIndex++) {
        int2 result = adjustOffset(xOldRange, xNewRange, maxLoc[imageIndex].x);
        maxLoc[imageIndex].x = result.x;
        maxLocShift[imageIndex].x = result.y;
        result = adjustOffset(yOldRange, yNewRange, maxLoc[imageIndex].y);
        maxLoc[imageIndex].y = result.x;
        maxLocShift[imageIndex].y = result.y;
    }
}

} // namespace
// end of file
