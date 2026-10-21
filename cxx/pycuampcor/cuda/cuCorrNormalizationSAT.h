/**
 * @file  cuCorrNormalizationSAT.h
 * @brief Normalization of correlation surfaces with sum area tables (CUDA backend)
 *
 * Implemented in cuCorrNormalizationSAT.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// normalize the correlation surfaces (in place) by the norms of the reference windows (with the
/// mean subtracted) and of the secondary windows under them, from the sum area tables of the
/// secondary windows
/// @param referenceSum2, secondarySAT, secondarySAT2 work arrays: the sums of the squares of the
///     reference windows (1 x 1 each), the sum area tables of the secondary windows and of their squares
void cuCorrNormalizeSAT(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
    ArraysView<const real_type> secondary, ArraysView<real_type> referenceSum2,
    ArraysView<real_type> secondarySAT, ArraysView<real_type> secondarySAT2, stream_t stream);

} // namespace
//end of file
