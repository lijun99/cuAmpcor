/**
 * @file  cuSurfaceOverSampler.h
 * @brief The interface of the oversamplers of correlation surfaces around their peaks
 *
 * The correlation surfaces around their peaks are oversampled to find the sub-pixel peak, with
 * FFT (the whole surface) or sinc interpolation (around the peak only). The packages (cpu, cuda)
 * implement both for their arrays, and create one by the method with newSurfaceOverSampler
 * (cuOverSampler.h of the package).
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"

namespace pycuampcor {

/// an oversampler of correlation surfaces around their peaks, for the arrays arrayT
template <template <typename> class arrayT>
class cuSurfaceOverSampler {
public:
    using real_type = typename array_traits<arrayT>::real_type;
    using int2_type = typename array_traits<arrayT>::int2_type;

    virtual ~cuSurfaceOverSampler() = default;

    /// oversample the correlation surfaces
    /// @param surfaces the correlation surfaces around their peaks
    /// @param[out] oversampled the oversampled surfaces
    /// @param shift the shift of each peak from the center of its surface (the sinc oversampler
    ///     computes only around the peak), or nullptr if the peaks are at the centers
    virtual void execute(arrayT<real_type> *surfaces, arrayT<real_type> *oversampled,
        arrayT<int2_type> *shift) = 0;
};

} // namespace
//end of file
