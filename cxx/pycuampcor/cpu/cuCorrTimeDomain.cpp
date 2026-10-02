/*
 * @file  cuCorrTimeDomain.cpp
 * @brief Correlation between two sets of images in time domain (CPU backend)
 *
 */

#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

/**
 * Perform cross correlation in time domain
 * @param[in] templates Reference images
 * @param[in] images Secondary images
 * @param[out] results Output correlation surface
 */
void cuCorrTimeDomain(cuArrays<real_type> *templates,
               cuArrays<real_type> *images,
               cuArrays<real_type> *results,
               stream_t)
{
    const int templateNX = templates->height, templateNY = templates->width;
    const int imageNY = images->width;
    const int resultNX = results->height, resultNY = results->width;

    for(int imageIdx = 0; imageIdx < images->count; imageIdx++) {
        const real_type *templateD = templates->devData + (size_t)imageIdx*templates->size;
        const real_type *imageD = images->devData + (size_t)imageIdx*images->size;
        real_type *resultD = results->devData + (size_t)imageIdx*results->size;
        // accumulate a row of the correlation surface at once: the innermost loop over
        // the (independent) result pixels vectorizes, while the summation order of each
        // result pixel is kept (over the template pixels, row by row)
        for(int rx = 0; rx < resultNX; rx++) {
            real_type *corrCoeff = resultD + rx*resultNY;
            for(int ry = 0; ry < resultNY; ry++)
                corrCoeff[ry] = 0;
            for(int tx = 0; tx < templateNX; tx++) {
                const real_type *t = templateD + tx*templateNY;
                const real_type *imgRow = imageD + (rx+tx)*imageNY;
                for(int ty = 0; ty < templateNY; ty++) {
                    const real_type tValue = t[ty];
                    const real_type *img = imgRow + ty;
                    for(int ry = 0; ry < resultNY; ry++)
                        corrCoeff[ry] += tValue*img[ry];
                }
            }
        }
    }
}

} // namespace
// end of file
