/**
 * @file  CorrTimeDomain.cpp
 * @brief Correlation between two sets of images in time domain (CPU backend)
 *
 */

#include "CorrTimeDomain.h"
#include "float2.h"

namespace pycuampcor::cpu {

/**
 * Perform cross correlation in time domain
 * @param[in] templates Reference images
 * @param[in] images Secondary images
 * @param[out] results Output correlation surface
 */
void cuCorrTimeDomain(ArraysView<const real_type> templates,
               ArraysView<const real_type> images,
               ArraysView<real_type> results,
               stream_t)
{
    const int templateNX = templates.height(), templateNY = templates.width();
    const int imageNY = images.width();
    const int resultNX = results.height(), resultNY = results.width();

    for(int imageIdx = 0; imageIdx < images.count(); imageIdx++) {
        const real_type *templateD = templates.array(imageIdx).data_handle();
        const real_type *imageD = images.array(imageIdx).data_handle();
        real_type *resultD = results.array(imageIdx).data_handle();
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
