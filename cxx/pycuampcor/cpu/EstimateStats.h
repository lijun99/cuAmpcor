/**
 * @file  EstimateStats.h
 * @brief SNR and covariance of the offsets (CPU backend)
 *
 * Implemented in EstimateStats.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuEstimateSnr(array_type<real_type> *corrSum, array_type<int> *corrValidCount,
    array_type<real_type> *maxval, array_type<real_type> *snrValue, stream_t stream);

void cuEstimateVariance(array_type<real_type> *corrBatchRaw, array_type<int2> *maxloc,
    array_type<real_type> *maxval, const int templateSize, const int distance,
    array_type<real3_type> *covValue, stream_t stream);

} // namespace
//end of file
