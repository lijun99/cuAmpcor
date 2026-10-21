/**
 * @file  cuDeramp.h
 * @brief Removal of the phase ramp of complex images (CPU backend)
 *
 * Implemented in cuDeramp.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuDeramp(DerampMethod method, cuArrays<complex_type> *images, DerampAxis axis,
    stream_t stream);

} // namespace
//end of file
