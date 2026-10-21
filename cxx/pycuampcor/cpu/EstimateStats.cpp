/**
 * @file  EstimateStats.cpp
 * @brief Estimate the statistics of the correlation surface (CPU backend)
 *
 * 9/23/2017, Minyan Zhong
 */

#include "EstimateStats.h"
#include "float2.h"

namespace pycuampcor::cpu {

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[in] maxval the peak values
 * @param[out] snrValue return snr value
 */
void cuEstimateSnr(ArraysView<const real_type> corrSum, ArraysView<const int> corrValidCount,
    ArraysView<const real_type> maxval, ArraysView<real_type> snrValue, stream_t)
{
    const int size = corrSum.size();
    for(int idx = 0; idx < size; idx++) {
        const real_type peak = maxval[idx];
        real_type mean = (corrSum[idx] - peak * peak) / (corrValidCount[idx] - 1);
        snrValue[idx] = peak * peak / mean;
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
void cuEstimateVariance(ArraysView<const real_type> corrBatchRaw, ArraysView<const int2> maxloc,
    ArraysView<const real_type> maxval, const int templateSize, const int distance,
    ArraysView<real3_type> covValue, stream_t)
{
    const int NX = corrBatchRaw.height();
    const int NY = corrBatchRaw.width();

    for(int idxImage = 0; idxImage < corrBatchRaw.count(); idxImage++) {
        int px = maxloc[idxImage].x;
        int py = maxloc[idxImage].y;
        real_type peak = maxval[idxImage];

        // Check if maxval is on the margin.
        if (px-distance < 0 || py-distance <0 || px + distance >=NX || py+distance >=NY)  {
            covValue[idxImage] = make_real3(99.0, 99.0, 0.0);
            continue;
        }

        // the surface at the peak and at distance pixels around it
        const ArraysView<const real_type> corr = corrBatchRaw.array(idxImage);
        const int x0 = px - distance, x2 = px + distance;
        const int y0 = py - distance, y2 = py + distance;

        // second-order derivatives
        real_type dxx = - ( corr(x2, py) + corr(x0, py) - 2.0*corr(px, py) );
        real_type dyy = - ( corr(px, y2) + corr(px, y0) - 2.0*corr(px, py) ) ;
        real_type dxy = ( corr(x2, y2) + corr(x0, y0) - corr(x2, y0) - corr(x0, y2) ) *0.25;

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
            covValue[idxImage] = make_real3(99.0, 99.0, 0.0);
        }
        else {
            real_type cov_xx = (- n2 * u * dyy + n4 * ( dyy*dyy + dxy*dxy) ) / u2;
            real_type cov_yy = (- n2 * u * dxx + n4 * ( dxx*dxx + dxy*dxy) ) / u2;
            real_type cov_xy = (  n2 * u * dxy - n4 * ( dxx + dyy ) * dxy ) / u2;
            covValue[idxImage] = make_real3(cov_xx, cov_yy, cov_xy);
        }
    }
}

} // namespace
// end of file
