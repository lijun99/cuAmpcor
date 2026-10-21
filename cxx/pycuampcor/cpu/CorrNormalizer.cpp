/**
 * @file  CorrNormalizer.cpp
 * @brief processors to normalize the correlation surface (CPU backend)
 *
 */

#include "CorrNormalizer.h"
#include "float2.h"
#include "CorrNormalizationSAT.h"
#include "Arrays.h"

namespace pycuampcor::cpu {

/**
 * The normalizer of the correlation surfaces: cuNormalizeSAT for all sizes
 */
std::unique_ptr<cuNormalizer<Arrays>>
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // the sum area table algorithm applies to all sizes
    return std::make_unique<cuNormalizeSAT>(secondaryNX, secondaryNY, count);
}

/**
 * Constructor, allocating the sum area tables of the secondary windows
 */
cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work arrays for secondary sum and sum square
    secondarySAT = array_type<double>(secondaryNX, secondaryNY, count);
    secondarySAT2 = array_type<double>(secondaryNX, secondaryNY, count);
}

/**
 * Release the work arrays
 */
cuNormalizeSAT::~cuNormalizeSAT() = default;

/**
 * Normalize the correlation surfaces with the sum area tables of the secondary windows
 */
void cuNormalizeSAT::execute(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, stream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary, secondarySAT, secondarySAT2, stream);
}

} // namespace
// end of file
