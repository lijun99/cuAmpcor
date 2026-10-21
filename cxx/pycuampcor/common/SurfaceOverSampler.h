/**
 * @file  SurfaceOverSampler.h
 * @brief The interface of the oversamplers of correlation surfaces around their peaks
 *
 * The correlation surfaces around their peaks are oversampled to find the sub-pixel peak, with
 * FFT (the whole surface) or sinc interpolation (around the peak only). The packages (cpu, cuda)
 * implement both for their arrays, and create one by the method with newSurfaceOverSampler
 * (OverSampler.h of the cpu package, cuOverSampler.h of the cuda package).
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"
#include "ArraysView.h"

namespace pycuampcor {

/// an oversampler of correlation surfaces around their peaks, for the arrays arrayT
template <template <typename> class arrayT>
class SurfaceOverSampler {
public:
    using real_type = typename array_traits<arrayT>::real_type;  ///< the real type
    using int2_type = typename array_traits<arrayT>::int2_type;  ///< a pair of ints (e.g., the peak shifts)

    virtual ~SurfaceOverSampler() = default;

    /// oversample the correlation surfaces
    /// @param surfaces the correlation surfaces around their peaks
    /// @param[out] oversampled the oversampled surfaces
    /// @param shift the shift of each peak from the center of its surface (the sinc oversampler
    ///     computes only around the peak), or empty if the peaks are at the centers
    virtual void execute(ArraysView<const real_type> surfaces, ArraysView<real_type> oversampled,
        ArraysView<const int2_type> shift) = 0;
};

} // namespace
//end of file
