/**
 * @file  cuEstimateStats.h
 * @brief SNR and covariance of the offsets (CUDA backend)
 *
 * Implemented in cuEstimateStats.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// the SNR of the correlation surfaces: the squared peak maxval over the mean of the squares of the
/// other corrValidCount - 1 pixels around it (corrSum the sum of the squares, with the peak's)
void cuEstimateSnr(ArraysView<const real_type> corrSum, ArraysView<const int> corrValidCount,
    ArraysView<const real_type> maxval, ArraysView<real_type> snrValue, stream_t stream);

/// the covariance of the offsets (xx, yy, xy) from the curvature of the correlation surfaces at
/// their peaks (maxloc, maxval), from the pixels at distance around the peak, for templates of
/// templateSize pixels; (99, 99, 0) for peaks at the edges or of too small curvatures
void cuEstimateVariance(ArraysView<const real_type> corrBatchRaw, ArraysView<const int2> maxloc,
    ArraysView<const real_type> maxval, const int templateSize, const int distance,
    ArraysView<real3_type> covValue, stream_t stream);

} // namespace
//end of file
