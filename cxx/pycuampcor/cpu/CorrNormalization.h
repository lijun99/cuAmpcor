/**
 * @file  CorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CPU backend)
 *
 * Implemented in CorrNormalization.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysSubtractMean(array_type<real_type> *images, stream_t stream);

void cuArraysSumCorr(array_type<real_type> *images, array_type<int> *imagesValid,
    array_type<real_type> *imagesSum, array_type<int> *imagesValidCount, stream_t stream);

} // namespace
//end of file
