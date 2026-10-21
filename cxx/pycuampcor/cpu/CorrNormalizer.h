/**
 * @file  CorrNormalizer.h
 * @brief The normalizers of correlation surfaces (CPU package; see Normalizer.h)
 *
 * The CPU package uses the sum area table based algorithm (cuNormalizeSAT) for all sizes.
 */

#pragma once

#include "forward.h"
#include "Arrays.h"
#include "traits.h"
#include "Normalizer.h"

namespace pycuampcor::cpu {

/// the normalizer of the correlation surfaces of count windows, with secondary windows of
/// secondaryNX x secondaryNY
std::unique_ptr<Normalizer<Arrays>> newNormalizer(int secondaryNX, int secondaryNY, int count);

/// the normalizer with sum area tables of the secondary windows (computed in double precision)
class cuNormalizeSAT : public Normalizer<Arrays> {
public:
    /// a normalizer with the sum area tables for count secondary windows of secondaryNX x secondaryNY
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    /// destructor
    ~cuNormalizeSAT() override;

    /// normalize the correlation surfaces (see Normalizer::execute)
    void execute(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
        ArraysView<const real_type> secondary, stream_t stream) override;

private:
    array_type<double> secondarySAT;    ///< the sum area tables of the secondary windows
    array_type<double> secondarySAT2;   ///< the same, of their squares
};

} // namespace
// end of file
