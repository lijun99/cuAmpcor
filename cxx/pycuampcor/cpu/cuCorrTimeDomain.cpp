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
        for(int rx = 0; rx < resultNX; rx++)
            for(int ry = 0; ry < resultNY; ry++) {
                real_type corrCoeff = 0;
                for(int tx = 0; tx < templateNX; tx++) {
                    const real_type *t = templateD + tx*templateNY;
                    const real_type *img = imageD + (rx+tx)*imageNY + ry;
                    for(int ty = 0; ty < templateNY; ty++)
                        corrCoeff += t[ty]*img[ty];
                }
                resultD[rx*resultNY+ry] = corrCoeff;
            }
    }
}

} // namespace
// end of file
