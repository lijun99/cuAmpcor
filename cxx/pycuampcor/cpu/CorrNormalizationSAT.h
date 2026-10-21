/**
 * @file  CorrNormalizationSAT.h
 * @brief Normalization of correlation surfaces with sum area tables (CPU backend)
 *
 * Implemented in CorrNormalizationSAT.cpp.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

void cuCorrNormalizeSAT(array_type<real_type> *correlation, array_type<real_type> *reference,
    array_type<real_type> *secondary, array_type<double> *secondarySAT, array_type<double> *secondarySAT2,
    stream_t stream);

} // namespace
//end of file
