/**
 * @file  cuCorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CUDA backend)
 *
 * Implemented in cuCorrTimeDomain.cu.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cuda {

void cuCorrTimeDomain(cuArrays<real_type> *templates, cuArrays<real_type> *images,
    cuArrays<real_type> *results, stream_t stream);

} // namespace
//end of file
