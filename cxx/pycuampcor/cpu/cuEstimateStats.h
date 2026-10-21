/**
 * @file  cuEstimateStats.h
 * @brief SNR and covariance of the offsets (CPU backend)
 *
 * Implemented in cuEstimateStats.cpp.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cpu {

void cuEstimateSnr(cuArrays<real_type> *corrSum, cuArrays<int> *corrValidCount,
    cuArrays<real_type> *maxval, cuArrays<real_type> *snrValue, stream_t stream);

void cuEstimateVariance(cuArrays<real_type> *corrBatchRaw, cuArrays<int2> *maxloc,
    cuArrays<real_type> *maxval, const int templateSize, const int distance,
    cuArrays<real3_type> *covValue, stream_t stream);

} // namespace
//end of file
