/**
 * @file  cuCorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CUDA backend)
 *
 * Implemented in cuCorrNormalization.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// subtract the mean of each array from its elements
void cuArraysSubtractMean(ArraysView<real_type> images, stream_t stream);

/// the sum of the squares of the elements of each array (imagesSum), and the number of its valid
/// elements (imagesValidCount, the sum of the flags imagesValid)
void cuArraysSumCorr(ArraysView<const real_type> images, ArraysView<const int> imagesValid,
    ArraysView<real_type> imagesSum, ArraysView<int> imagesValidCount, stream_t stream);

/// normalize the correlation surfaces (in place) by the norms of the reference windows (with the
/// mean subtracted) and of the secondary windows under them, with Size threads per window (one per
/// column of the secondary windows, of up to Size columns)
template<int Size>
void cuCorrNormalizeFixed(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
    ArraysView<const real_type> secondary, stream_t stream);

} // namespace
//end of file
