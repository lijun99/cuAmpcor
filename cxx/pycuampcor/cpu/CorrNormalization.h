/**
 * @file  CorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CPU backend)
 *
 * Implemented in CorrNormalization.cpp.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cpu {

/// subtract the mean of each array from its elements
void cuArraysSubtractMean(ArraysView<real_type> images, stream_t stream);

/// the sum of the squares of the elements of each array (imagesSum), and the number of its valid
/// elements (imagesValidCount, the sum of the flags imagesValid)
void cuArraysSumCorr(ArraysView<const real_type> images, ArraysView<const int> imagesValid,
    ArraysView<real_type> imagesSum, ArraysView<int> imagesValidCount, stream_t stream);

} // namespace
//end of file
