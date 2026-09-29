/**
 * @file  cuEstimateStats.cpp
 * @brief Estimate the statistics of the correlation surface (CPU backend)
 *
 * 9/23/2017, Minyan Zhong
 */

#include "cuAmpcorUtil.h"
#include <algorithm>
#include <cmath>

namespace pycuampcor::cpu {

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] maxval the peak values
 * @param[out] snrValue return snr value
 * @param[in] size the number of pixels contributing to sum
 */
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, const int size, stream_t)
{
    const size_t nImages = corrSum->getSize();
    for(size_t idx = 0; idx < nImages; idx++) {
        real_type peak = maxval->devData[idx];
        peak *= peak;
        real_type mean = (corrSum->devData[idx] - peak) / (size - 1);
        snrValue->devData[idx] = peak / mean;
    }
}

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[in] maxval the peak values
 * @param[out] snrValue return snr value
 */
void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<int> *corrValidCount, cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, stream_t)
{
    const size_t size = corrSum->getSize();
    for(size_t idx = 0; idx < size; idx++) {
        const real_type peak = maxval->devData[idx];
        real_type mean = (corrSum->devData[idx] - peak * peak) / (corrValidCount->devData[idx] - 1);
        snrValue->devData[idx] = peak * peak / mean;
    }
}

/**
 * Estimate the variance of the correlation surface
 * @param[in] corrBatchRaw correlation surface
 * @param[in] maxloc maximum location
 * @param[in] maxval maximum value
 * @param[in] templateSize size of reference chip
 * @param[in] distance distance between a pixel
 * @param[out] covValue variance value
 */
void cuEstimateVariance(cuArrays<real_type> *corrBatchRaw, cuArrays<int2> *maxloc, cuArrays<real_type> *maxval, const int templateSize, const int distance, cuArrays<real3_type> *covValue, stream_t)
{
    const int NX = corrBatchRaw->height;
    const int NY = corrBatchRaw->width;
    const real_type *corr = corrBatchRaw->devData;

    for(int idxImage = 0; idxImage < corrBatchRaw->count; idxImage++) {
        int px = maxloc->devData[idxImage].x;
        int py = maxloc->devData[idxImage].y;
        real_type peak = maxval->devData[idxImage];

        // Check if maxval is on the margin.
        if (px-distance < 0 || py-distance <0 || px + distance >=NX || py+distance >=NY)  {
            covValue->devData[idxImage] = make_real3(99.0, 99.0, 0.0);
            continue;
        }

        int offset = NX * NY * idxImage;
        int idx00 = offset + (px - distance) * NY + py - distance;
        int idx01 = offset + (px - distance) * NY + py    ;
        int idx02 = offset + (px - distance) * NY + py + distance;
        int idx10 = offset + (px    ) * NY + py - distance;
        int idx11 = offset + (px    ) * NY + py    ;
        int idx12 = offset + (px    ) * NY + py + distance;
        int idx20 = offset + (px + distance) * NY + py - distance;
        int idx21 = offset + (px + distance) * NY + py    ;
        int idx22 = offset + (px + distance) * NY + py + distance;

        // second-order derivatives
        real_type dxx = - ( corr[idx21] + corr[idx01] - 2.0*corr[idx11] );
        real_type dyy = - ( corr[idx12] + corr[idx10] - 2.0*corr[idx11] ) ;
        real_type dxy = ( corr[idx22] + corr[idx00] - corr[idx20] - corr[idx02] ) *0.25;

        real_type n2 = std::max(1.0 - peak, 0.0);

        dxx = dxx * templateSize;
        dyy = dyy * templateSize;
        dxy = dxy * templateSize;

        real_type n4 = n2*n2;
        n2 = n2 * 2;
        n4 = n4 * 0.5 * templateSize;

        real_type u = dxy * dxy - dxx * dyy;
        real_type u2 = u*u;

        // if the Gaussian curvature is too small
        if (std::fabs(u) < 1e-2) {
            covValue->devData[idxImage] = make_real3(99.0, 99.0, 0.0);
        }
        else {
            real_type cov_xx = (- n2 * u * dyy + n4 * ( dyy*dyy + dxy*dxy) ) / u2;
            real_type cov_yy = (- n2 * u * dxx + n4 * ( dxx*dxx + dxy*dxy) ) / u2;
            real_type cov_xy = (  n2 * u * dxy - n4 * ( dxx + dyy ) * dxy ) / u2;
            covValue->devData[idxImage] = make_real3(cov_xx, cov_yy, cov_xy);
        }
    }
}

} // namespace
// end of file
