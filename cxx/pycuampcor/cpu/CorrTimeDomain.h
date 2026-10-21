/**
 * @file  CorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CPU backend)
 *
 * Implemented in CorrTimeDomain.cpp.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cpu {

/// cross-correlate the templates (reference windows) with the images (secondary windows) in the
/// time domain: the k-th surface of results, of (image - template + 1) pixels each way, from the k-th
/// template and image
void cuCorrTimeDomain(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results, stream_t stream);

} // namespace
//end of file
