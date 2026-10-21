/**
 * @file  cuDeramp.h
 * @brief Removal of the phase ramp of complex images (CUDA backend)
 *
 * Implemented in cuDeramp.cu.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cuda {

void cuDeramp(DerampMethod method, cuArrays<complex_type> *images, DerampAxis axis,
    stream_t stream);

} // namespace
//end of file
