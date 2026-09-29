/*
 * @file cuCorrNormalizer.cpp
 * @brief processors to normalize the correlation surface (CPU backend)
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

cuNormalizeProcessor*
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // the sum area table algorithm applies to all sizes
    return new cuNormalizeSAT(secondaryNX, secondaryNY, count);
}

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work arrays for secondary sum and sum square
    secondarySAT = new cuArrays<double>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = new cuArrays<double>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
}

cuNormalizeSAT::~cuNormalizeSAT()
{
    delete secondarySAT;
    delete secondarySAT2;
}

void cuNormalizeSAT::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, stream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        secondarySAT, secondarySAT2, stream);
}

} // namespace
// end of file
