/*
 * @file cuCorrNormalizer.cpp
 * @brief processors to normalize the correlation surface (CPU backend)
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

std::unique_ptr<cuNormalizeProcessor>
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // the sum area table algorithm applies to all sizes
    return std::make_unique<cuNormalizeSAT>(secondaryNX, secondaryNY, count);
}

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work arrays for secondary sum and sum square
    secondarySAT = std::make_unique<cuArrays<double>>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = std::make_unique<cuArrays<double>>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
}

cuNormalizeSAT::~cuNormalizeSAT() = default;

void cuNormalizeSAT::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, stream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        secondarySAT.get(), secondarySAT2.get(), stream);
}

} // namespace
// end of file
