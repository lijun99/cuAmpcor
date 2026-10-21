/**
 * @file  cuEstimateStats.cu
 * @brief Estimate the statistics of the correlation surface
 *
 * 9/23/2017, Minyan Zhong
 */

#include "cuEstimateStats.h"
#include "float2.h"
#include "device.h"
#include "cudaError.h"
#include "float.h"

namespace pycuampcor::cuda {

__global__ void cudaKernel_estimateSnr(ArraysView<const real_type> corrSum,
    ArraysView<const int> corrValidCount, ArraysView<const real_type> maxval,
    ArraysView<real_type> snrValue, const int size)

{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= size) return;

    real_type mean = (corrSum[idx] - maxval[idx] * maxval[idx]) / (corrValidCount[idx] - 1);

    snrValue[idx] = maxval[idx] * maxval[idx] / mean;
}

/**
 * Estimate the signal to noise ratio (SNR) of the correlation surface
 * @param[in] corrSum the sum of the correlation surface
 * @param[in] corrValidCount the number of valid pixels contributing to sum
 * @param[out] snrValue return snr value
 * @param[in] stream cuda stream
 */
void cuEstimateSnr(ArraysView<const real_type> corrSum, ArraysView<const int> corrValidCount,
    ArraysView<const real_type> maxval, ArraysView<real_type> snrValue, cudaStream_t stream)
{

    int size = corrSum.size();
    cudaKernel_estimateSnr<<< divUp(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrSum, corrValidCount, maxval, snrValue, size);
    checkCudaAsyncErrors();
}

// cuda kernel for cuEstimateVariance
__global__ void cudaKernel_estimateVar(ArraysView<const real_type> corrBatchRaw,
    ArraysView<const int2> maxloc, ArraysView<const real_type> maxval, const int templateSize,
    const int distance, ArraysView<real3_type> covValue, const int size)
{
    const int NX = corrBatchRaw.height(), NY = corrBatchRaw.width();

    // Find image id.
    int idxImage = threadIdx.x + blockDim.x*blockIdx.x;

    if (idxImage >= size) return;

    // Preparation.
    int px = maxloc[idxImage].x;
    int py = maxloc[idxImage].y;
    real_type peak = maxval[idxImage];

    // Check if maxval is on the margin.
    if (px-distance < 0 || py-distance <0 || px + distance >=NX || py+distance >=NY)  {

        covValue[idxImage] = make_real3(99.0, 99.0, 0.0);

    }
    else {
        // the surface at the peak and at distance pixels around it
        const ArraysView<const real_type> corr = corrBatchRaw.array(idxImage);
        const int x0 = px - distance, x2 = px + distance;
        const int y0 = py - distance, y2 = py + distance;

        // second-order derivatives
        real_type dxx = - ( corr(x2, py) + corr(x0, py) - 2.0*corr(px, py) );
        real_type dyy = - ( corr(px, y2) + corr(px, y0) - 2.0*corr(px, py) ) ;
        real_type dxy = ( corr(x2, y2) + corr(x0, y0) - corr(x2, y0) - corr(x0, y2) ) *0.25;

        real_type n2 = max(1.0 - peak, 0.0);

        dxx = dxx * templateSize;
        dyy = dyy * templateSize;
        dxy = dxy * templateSize;

        real_type n4 = n2*n2;
        n2 = n2 * 2;
        n4 = n4 * 0.5 * templateSize;

        real_type u = dxy * dxy - dxx * dyy;
        real_type u2 = u*u;

        // if the Gaussian curvature is too small
        if (fabsf(u) < 1e-2) {
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

/**
 * Estimate the variance of the correlation surface
 * @param[in] templateSize size of reference chip
 * @param[in] distance distance between a pixel
 * @param[in] corrBatchRaw correlation surface
 * @param[in] maxloc maximum location
 * @param[in] maxval maximum value
 * @param[out] covValue variance value
 * @param[in] stream cuda stream
 */
void cuEstimateVariance(ArraysView<const real_type> corrBatchRaw, ArraysView<const int2> maxloc,
    ArraysView<const real_type> maxval, const int templateSize, const int distance,
    ArraysView<real3_type> covValue, cudaStream_t stream)
{
    int size = corrBatchRaw.count();
    // One dimensional launching parameters to loop over every correlation surface.
    cudaKernel_estimateVar<<< divUp(size, NTHREADS), NTHREADS, 0, stream>>>
        (corrBatchRaw, maxloc, maxval, templateSize, distance, covValue, size);
    checkCudaAsyncErrors();
}
//end of file

} // namespace