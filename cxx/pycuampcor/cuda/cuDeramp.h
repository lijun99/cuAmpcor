/**
 * @file  cuDeramp.h
 * @brief Removal of the phase ramp of complex images (CUDA backend)
 *
 * Implemented in cuDeramp.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// remove the phase ramps of the complex windows (in place), along the axis, with the method
/// (only DerampMethod::Complex deramps; the other methods leave the windows as they are)
void cuDeramp(DerampMethod method, ArraysView<complex_type> images, DerampAxis axis,
    stream_t stream);

} // namespace
//end of file
