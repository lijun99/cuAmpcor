/**
 * @file  cuCorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CPU backend)
 *
 * Implemented in cuCorrTimeDomain.cpp.
 */

#pragma once

#include "backend.h"
#include "cuArrays.h"
#include "data_types.h"

namespace pycuampcor::cpu {

void cuCorrTimeDomain(cuArrays<real_type> *templates, cuArrays<real_type> *images,
    cuArrays<real_type> *results, stream_t stream);

} // namespace
//end of file
