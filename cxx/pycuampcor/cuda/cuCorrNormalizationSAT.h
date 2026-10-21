/**
 * @file  cuCorrNormalizationSAT.h
 * @brief Normalization of correlation surfaces with sum area tables (CUDA backend)
 *
 * Implemented in cuCorrNormalizationSAT.cu.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cuda {

void cuCorrNormalizeSAT(cuArrays<real_type> *correlation, cuArrays<real_type> *reference,
    cuArrays<real_type> *secondary, cuArrays<real_type> * referenceSum2,
    cuArrays<real_type> *secondarySAT, cuArrays<real_type> *secondarySAT2, stream_t stream);

} // namespace
//end of file
