/*
 * @file cuCorrNormalizationSAT.cpp
 * @brief Utilities to normalize the 2D correlation surface with the sum area table (CPU backend)
 *
 */

#include "cuAmpcorUtil.h"
#include <cmath>

namespace pycuampcor::cpu {

/**
 * Normalize the correlation surface with the sum area table
 * @param[in,out] correlation un-normalized correlation surface as input and normalized as output
 * @param[in] reference reference windows with mean subtracted
 * @param[in] secondary secondary (search) windows
 * @param[out] secondarySat work array for the sum area table of the secondary windows
 * @param[out] secondarySat2 work array for the sum area table of the secondary windows squared
 */
void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary,
    cuArrays<double> *secondarySat, cuArrays<double> *secondarySat2, stream_t)
{
    const int corNX = correlation->height, corNY = correlation->width;
    const int referenceNX = reference->height, referenceNY = reference->width;
    const int secondaryNX = secondary->height, secondaryNY = secondary->width;
    const int referenceSize = reference->size;
    const int secondarySize = secondary->size;

    for(int imageid = 0; imageid < correlation->count; imageid++) {
        // compute the sum of value^2 of the reference image
        // note that the mean is already subtracted
        const real_type *ref = reference->devData + (size_t)imageid*referenceSize;
        double refSum2 = 0.0;
        for(int i = 0; i < referenceSize; i++)
            refSum2 += ref[i]*ref[i];

        // compute the (inclusive) sum area tables of the secondary image, for value and value^2
        const real_type *data = secondary->devData + (size_t)imageid*secondarySize;
        double *sat = secondarySat->devData + (size_t)imageid*secondarySize;
        double *sat2 = secondarySat2->devData + (size_t)imageid*secondarySize;
        for(int row = 0; row < secondaryNX; row++) {
            double sum = 0.0, sum2 = 0.0;
            for(int col = 0; col < secondaryNY; col++) {
                const int index = row*secondaryNY + col;
                const double val = data[index];
                sum += val;
                sum2 += val*val;
                sat[index] = (row > 0) ? sat[index-secondaryNY] + sum : sum;
                sat2[index] = (row > 0) ? sat2[index-secondaryNY] + sum2 : sum2;
            }
        }

        // area sum of a referenceNX x referenceNY window starting at (tx, ty)
        auto areaSum = [&](const double *s, int tx, int ty) {
            double topleft = (tx > 0 && ty > 0) ? s[(tx-1)*secondaryNY+(ty-1)] : 0.0;
            double topright = (tx > 0 ) ? s[(tx-1)*secondaryNY+(ty+referenceNY-1)] : 0.0;
            double bottomleft = (ty > 0) ? s[(tx+referenceNX-1)*secondaryNY+(ty-1)] : 0.0;
            double bottomright = s[(tx+referenceNX-1)*secondaryNY+(ty+referenceNY-1)];
            return bottomright + topleft - topright - bottomleft;
        };

        // normalize the correlation surface
        real_type *corr = correlation->devData + (size_t)imageid*corNX*corNY;
        for(int tx = 0; tx < corNX; tx++)
            for(int ty = 0; ty < corNY; ty++) {
                const double secondarySum = areaSum(sat, tx, ty);
                const double secondarySum2 = areaSum(sat2, tx, ty);
                const double norm2 = (secondarySum2-secondarySum*secondarySum/referenceSize)*refSum2;
                corr[tx*corNY+ty] *= 1.0/std::sqrt(norm2 + EPSILON);
            }
    }
}

} // namespace
// end of file
