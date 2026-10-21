/**
 * @file  CorrNormalizationSAT.h
 * @brief Normalization of correlation surfaces with sum area tables (CPU backend)
 *
 * Implemented in CorrNormalizationSAT.cpp.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cpu {

/// normalize the correlation surfaces (in place) by the norms of the reference windows (with the
/// mean subtracted) and of the secondary windows under them, from the sum area tables of the
/// secondary windows
/// @param secondarySAT,secondarySAT2 work arrays: the sum area tables of the secondary windows and
///     of their squares
void cuCorrNormalizeSAT(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
    ArraysView<const real_type> secondary, ArraysView<double> secondarySAT,
    ArraysView<double> secondarySAT2, stream_t stream);

} // namespace
//end of file
