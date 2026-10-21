/*
 * @file cuNormalizer.cu
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

namespace pycuampcor::cuda {

std::unique_ptr<cuNormalizeProcessor>
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

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = std::make_unique<cuArrays<real_type>>(1, 1, count);
    referenceSum2->allocate();

    // secondary sum and sum square
    secondarySAT = std::make_unique<cuArrays<real_type>>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = std::make_unique<cuArrays<real_type>>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
};

cuNormalizeSAT::~cuNormalizeSAT() = default;

void cuNormalizeSAT::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2.get(), secondarySAT.get(), secondarySAT2.get(), stream);
}

template<int Size>
void cuNormalizeFixed<Size>::execute(cuArrays<real_type> *correlation,
    cuArrays<real_type> *reference, cuArrays<real_type> *secondary, cudaStream_t stream)
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