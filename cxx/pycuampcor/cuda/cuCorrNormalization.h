/**
 * @file  cuCorrNormalization.h
 * @brief Statistics and normalization of correlation surfaces (CUDA backend)
 *
 * Implemented in cuCorrNormalization.cu.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cuda {

void cuArraysSubtractMean(cuArrays<real_type> *images, stream_t stream);

void cuCorrNormalize(cuArrays<real_type> *templates, cuArrays<real_type> *images,
    cuArrays<real_type> *results, stream_t stream);

template<int Size>
void cuCorrNormalizeFixed(cuArrays<real_type> *correlation, cuArrays<real_type> *reference,
    cuArrays<real_type> *secondary, stream_t stream);

void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid,
    cuArrays<real_type> *imagesSum, cuArrays<int> *imagesValidCount, stream_t stream);

} // namespace
//end of file
