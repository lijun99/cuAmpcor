/**
 * @file  CorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CPU backend)
 *
 * Implemented in CorrTimeDomain.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuCorrTimeDomain(array_type<real_type> *templates, array_type<real_type> *images,
    array_type<real_type> *results, stream_t stream);

} // namespace
//end of file
