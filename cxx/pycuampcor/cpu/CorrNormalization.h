/**
 * @file  CorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CPU backend)
 *
 * Implemented in CorrNormalization.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuArraysSubtractMean(cuArrays<real_type> *images, stream_t stream);

void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid,
    cuArrays<real_type> *imagesSum, cuArrays<int> *imagesValidCount, stream_t stream);

} // namespace
//end of file
