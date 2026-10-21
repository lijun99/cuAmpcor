/**
 * @file  cuCorrNormalizationSAT.h
 * @brief Normalization of correlation surfaces with sum area tables (CPU backend)
 *
 * Implemented in cuCorrNormalizationSAT.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference,
    cuArrays<real_type> *secondary, cuArrays<double> *secondarySAT, cuArrays<double> *secondarySAT2,
    stream_t stream);

} // namespace
//end of file
