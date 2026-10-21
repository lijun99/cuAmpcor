/**
 * @file  cuCorrNormalizer.h
 * @brief The normalizers of correlation surfaces (CUDA package; see Normalizer.h)
 *
 * cuNormalizeFixed<64/128/.../1024> sums in shared memory, with as many threads in a block as the
 * size; cuNormalizeSAT uses sum area tables, for any size (used beyond 1024).
 */

#pragma once

#include "forward.h"
#include "cuArrays.h"
#include "traits.h"
#include "Normalizer.h"

namespace pycuampcor::cuda {

/// the normalizer of the correlation surfaces of count windows, with secondary windows of
/// secondaryNX x secondaryNY: cuNormalizeFixed for the smallest size that fits, or cuNormalizeSAT
std::unique_ptr<Normalizer<cuArrays>> newNormalizer(int secondaryNX, int secondaryNY, int count);

/// the normalizer summing in shared memory, for windows of up to Size pixels along a side
template <int Size>
class cuNormalizeFixed : public Normalizer<cuArrays> {
public:
    /// normalize the correlation surfaces (see Normalizer::execute)
    void execute(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
        ArraysView<const real_type> secondary, cudaStream_t stream) override;
};

/// the normalizer with sum area tables of the secondary windows
class cuNormalizeSAT : public Normalizer<cuArrays> {
public:
    /// a normalizer with the sum area tables for count secondary windows of secondaryNX x secondaryNY
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    /// destructor
    ~cuNormalizeSAT() override;

    /// normalize the correlation surfaces (see Normalizer::execute)
    void execute(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
        ArraysView<const real_type> secondary, cudaStream_t stream) override;

private:
    cuArrays<real_type> referenceSum2;   ///< the sums of the squares of the reference windows
    cuArrays<real_type> secondarySAT;    ///< the sum area tables of the secondary windows
    cuArrays<real_type> secondarySAT2;   ///< the same, of their squares
};

} // namespace
// end of file
