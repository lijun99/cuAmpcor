/**
 * @file  Normalizer.h
 * @brief The interface of the normalizers of correlation surfaces
 *
 * The correlation surfaces of the windows are normalized by the norms of the reference window
 * and of the secondary window under it. The packages (cpu, cuda) implement it for their arrays,
 * with sum area tables or, for small windows on the GPU, sums in shared memory, and create the
 * implementation for a window size with newNormalizer (CorrNormalizer.h of the cpu package,
 * cuCorrNormalizer.h of the cuda package).
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"
#include "ArraysView.h"

namespace pycuampcor {

/// a normalizer of correlation surfaces, for the arrays arrayT
template <template <typename> class arrayT>
class Normalizer {
public:
    using real_type = typename array_traits<arrayT>::real_type;      ///< the real type
    using stream_type = typename array_traits<arrayT>::stream_type;  ///< the stream on which the kernels are queued

    virtual ~Normalizer() = default;

    /// normalize the correlation surfaces
    /// @param[in,out] correlation the correlation surfaces of the windows
    /// @param reference the reference windows (with the mean subtracted)
    /// @param secondary the secondary windows
    /// @param stream the stream on which the kernels are queued
    virtual void execute(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
        ArraysView<const real_type> secondary, stream_type stream) = 0;
};

} // namespace
//end of file
