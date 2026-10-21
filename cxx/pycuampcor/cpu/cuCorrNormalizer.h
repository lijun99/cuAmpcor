/**
 * @file  cuCorrNormalizer.h
 * @brief The normalizers of correlation surfaces (CPU package; see cuNormalizer.h)
 *
 * The CPU package uses the sum area table based algorithm (cuNormalizeSAT) for all sizes.
 */

#pragma once

#include "forward.h"
#include "arrayTraits.h"
#include "cuNormalizer.h"

namespace pycuampcor::cpu {

/// the normalizer of the correlation surfaces of count windows, with secondary windows of
/// secondaryNX x secondaryNY
std::unique_ptr<cuNormalizer<cuArrays>> newCuNormalizer(int secondaryNX, int secondaryNY, int count);

/// the normalizer with sum area tables of the secondary windows (computed in double precision)
class cuNormalizeSAT : public cuNormalizer<cuArrays> {
public:
    /// a normalizer with the sum area tables for count secondary windows of secondaryNX x secondaryNY
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    ~cuNormalizeSAT() override;

    /// normalize the correlation surfaces (see cuNormalizer::execute)
    void execute(cuArrays<real_type> *correlation, cuArrays<real_type> *reference,
        cuArrays<real_type> *secondary, stream_t stream) override;

private:
    std::unique_ptr<cuArrays<double>> secondarySAT;    ///< the sum area tables of the secondary windows
    std::unique_ptr<cuArrays<double>> secondarySAT2;   ///< the same, of their squares
};

} // namespace
// end of file
