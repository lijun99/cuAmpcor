/**
 * @file  cuCorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CPU backend)
 *
 * Implemented in cuCorrNormalization.cpp.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cpu {

void cuArraysSubtractMean(cuArrays<real_type> *images, stream_t stream);

void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid,
    cuArrays<real_type> *imagesSum, cuArrays<int> *imagesValidCount, stream_t stream);

} // namespace
//end of file
