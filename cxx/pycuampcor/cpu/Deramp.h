/**
 * @file  Deramp.h
 * @brief Removal of the phase ramp of complex images (CPU backend)
 *
 * Implemented in Deramp.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuDeramp(DerampMethod method, array_type<complex_type> *images, DerampAxis axis,
    stream_t stream);

} // namespace
//end of file
