/*
 * @file cuOffset.cpp
 * @brief Utilities used to determine the offset field (CPU backend)
 *
 */

// my module dependencies
#include "cuAmpcorUtil.h"

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
void cuArraysMaxloc2D(cuArrays<real_type> *images,
                      const int2 start, const int2 range,
                      cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, stream_t)
{
    const int imageNY = images->width;
    for(int imageIdx = 0; imageIdx < images->count; imageIdx++) {
        const real_type *image = images->devData + (size_t)imageIdx*images->size;
        real_type val = -REAL_MAX;
        int2 loc = start;
        for(int idx = start.x; idx < start.x + range.x; idx++)
            for(int idy = start.y; idy < start.y + range.y; idy++) {
                real_type pixelValue = image[IDX2R(idx, idy, imageNY)];
                if(val < pixelValue) {
                    val = pixelValue;
                    loc = make_int2(idx, idy);
                }
            }
        maxloc->devData[imageIdx] = loc;
        maxval->devData[imageIdx] = val;
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxloc arrays to hold the max locations
 * @param[out] maxval arrays to hold the max values
 */
void cuArraysMaxloc2D(cuArrays<real_type> *images,
                      cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, stream_t stream)
{
    // if no start and range are provided, use the whole image
    cuArraysMaxloc2D(images, make_int2(0, 0), make_int2(images->height, images->width),
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
void cuSubPixelOffset(cuArrays<int2> *offsetInit,
    cuArrays<int2> *offsetZoomIn,
    cuArrays<real2_type> *offsetFinal,
    const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor,
    stream_t)
{
    const size_t size = offsetInit->getSize();
    const real_type initRatio = 1.0f/(real_type)(initFactor);
    const real_type zoomInRatio = 1.0f/(real_type)(zoomInFactor);
    for(size_t idx = 0; idx < size; idx++) {
        const int2 init = offsetInit->devData[idx];
        const int2 zoomIn = offsetZoomIn->devData[idx];
        offsetFinal->devData[idx].x = initRatio*(init.x-initOrigin.x) + zoomInRatio*(zoomIn.x - zoomInOrigin.x);
        offsetFinal->devData[idx].y = initRatio*(init.y-initOrigin.y) + zoomInRatio*(zoomIn.y - zoomInOrigin.y);
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
void cuSubPixelOffset2Pass(cuArrays<int2> *offsetInit, cuArrays<int2> *offsetZoomIn,
    cuArrays<real2_type> *offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit,
    stream_t)
{
    const size_t size = offsetInit->getSize();
    const float OSratio = 1.0f/(float)(OverSampleRatioZoomin*OverSampleRatioRaw);
    const float xoffset = xHalfRangeInit ;
    const float yoffset = yHalfRangeInit ;
    for(size_t idx = 0; idx < size; idx++) {
        offsetFinal->devData[idx].x = OSratio*(offsetZoomIn->devData[idx].x) + offsetInit->devData[idx].x - xoffset;
        offsetFinal->devData[idx].y = OSratio*(offsetZoomIn->devData[idx].y) + offsetInit->devData[idx].y - yoffset;
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
void cuDetermineSecondaryExtractOffset(cuArrays<int2> *maxLoc, cuArrays<int2> *maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, stream_t)
{
    for(int imageIndex = 0; imageIndex < maxLoc->size; imageIndex++) {
        int2 result = adjustOffset(xOldRange, xNewRange, maxLoc->devData[imageIndex].x);
        maxLoc->devData[imageIndex].x = result.x;
        maxLocShift->devData[imageIndex].x = result.y;
        result = adjustOffset(yOldRange, yNewRange, maxLoc->devData[imageIndex].y);
        maxLoc->devData[imageIndex].y = result.x;
        maxLocShift->devData[imageIndex].y = result.y;
    }
}

} // namespace
// end of file
