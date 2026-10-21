/**
 * @file  cuCorrNormalizer.cpp
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "float2.h"
#include "cuCorrNormalization.h"
#include "cuCorrNormalizationSAT.h"
#include "cuArrays.h"

namespace pycuampcor::cuda {

/**
 * The normalizer of the correlation surfaces: cuNormalizeFixed for the smallest size that fits
 * the width of the secondary windows, or cuNormalizeSAT beyond 1024
 */
std::unique_ptr<cuNormalizer<cuArrays>>
newCuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // depending on NY, choose different processor
    if(secondaryNY <= 64) {
        return std::make_unique<cuNormalizeFixed<64>>();
    }
    else if (secondaryNY <= 128) {
        return std::make_unique<cuNormalizeFixed<128>>();
    }
    else if (secondaryNY <= 256) {
        return std::make_unique<cuNormalizeFixed<256>>();
    }
    else if (secondaryNY <= 512) {
        return std::make_unique<cuNormalizeFixed<512>>();
    }
    else if (secondaryNY <= 1024) {
        return std::make_unique<cuNormalizeFixed<1024>>();
    }
    else {
        return std::make_unique<cuNormalizeSAT>(secondaryNX, secondaryNY, count);
    }
}

/**
 * Constructor, allocating the sums of the squares of the reference windows and the sum area
 * tables of the secondary windows
 */
cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = cuArrays<real_type>(1, 1, count);

    // secondary sum and sum square
    secondarySAT = cuArrays<real_type>(secondaryNX, secondaryNY, count);
    secondarySAT2 = cuArrays<real_type>(secondaryNX, secondaryNY, count);
};

/**
 * Release the work arrays
 */
cuNormalizeSAT::~cuNormalizeSAT() = default;

/**
 * Normalize the correlation surfaces with the sum area tables of the secondary windows
 */
void cuNormalizeSAT::execute(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2, secondarySAT, secondarySAT2, stream);
}

/**
 * Normalize the correlation surfaces, summing in shared memory (blocks of Size threads)
 */
template<int Size>
void cuNormalizeFixed<Size>::execute(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream)
{
    cuCorrNormalizeFixed<Size>(correlation, reference, secondary, stream);
}

template class cuNormalizeFixed<64>;
template class cuNormalizeFixed<128>;
template class cuNormalizeFixed<256>;
template class cuNormalizeFixed<512>;
template class cuNormalizeFixed<1024>;

// end of file

} // namespace