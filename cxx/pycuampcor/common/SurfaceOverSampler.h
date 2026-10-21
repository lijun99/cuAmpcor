/**
 * @file  SurfaceOverSampler.h
 * @brief The interface of the oversamplers of correlation surfaces around their peaks
 *
 * The correlation surfaces around their peaks are oversampled to find the sub-pixel peak, with
 * FFT (the whole surface) or sinc interpolation (around the peak only), chosen at run time:
 * SurfaceOverSampler is the interface, implemented for both methods, for the arrays of any
 * package, by the oversamplers of real arrays on their device (OverSampler<method, real_type,
 * device>, see OverSampler.h); newSurfaceOverSampler creates one by the method. The
 * implementations are instantiated with the processors (the packages' processors.cpp), where the
 * oversamplers of the package are defined.
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"
#include "ArraysView.h"
#include "OverSampler.h"
#include "Options.h"

#include <memory>

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

/// the FFT oversampler of correlation surfaces (the whole surfaces), for the arrays arrayT
template <template <typename> class arrayT>
class FFTSurfaceOverSampler : public SurfaceOverSampler<arrayT> {
public:
    using traits = array_traits<arrayT>;                       ///< the types that go with the arrays
    using real_type = typename traits::real_type;              ///< the real type
    using int2_type = typename traits::int2_type;              ///< a pair of ints (the peak shifts)
    using stream_type = typename traits::stream_type;          ///< the stream of the work

    /// an oversampler of count surfaces of height x width, by factor
    FFTSurfaceOverSampler(int height, int width, int factor, int count, stream_type stream)
        : oversampler(height, width, height*factor, width*factor, count, stream) {}

    /// oversample the surfaces (the peaks need no shift: the whole surfaces are oversampled)
    void execute(ArraysView<const real_type> surfaces, ArraysView<real_type> oversampled,
        ArraysView<const int2_type>) override
    {
        oversampler.execute(surfaces, oversampled);
    }

private:
    OverSampler<OversamplingMethod::FFT, real_type, traits::device> oversampler;   ///< the FFT oversampler
};

/// the sinc oversampler of correlation surfaces (around their peaks only), for the arrays arrayT
template <template <typename> class arrayT>
class SincSurfaceOverSampler : public SurfaceOverSampler<arrayT> {
public:
    using traits = array_traits<arrayT>;                       ///< the types that go with the arrays
    using real_type = typename traits::real_type;              ///< the real type
    using int2_type = typename traits::int2_type;              ///< a pair of ints (the peak shifts)
    using stream_type = typename traits::stream_type;          ///< the stream of the work

    /// an oversampler by factor; the peaks are shifted by factor*rawFactor
    SincSurfaceOverSampler(int factor, int rawFactor, stream_type stream)
        : oversampler(factor, stream), shiftFactor(factor*rawFactor) {}

    /// oversample the surfaces around their peaks, shifted from the centers by shift (or not, if empty)
    void execute(ArraysView<const real_type> surfaces, ArraysView<real_type> oversampled,
        ArraysView<const int2_type> shift) override
    {
        if (!shift.empty())
            oversampler.execute(surfaces, oversampled, shift, shiftFactor);
        else
            oversampler.execute(surfaces, oversampled, traits::make_int2(0, 0), shiftFactor);
    }

private:
    OverSampler<OversamplingMethod::Sinc, real_type, traits::device> oversampler;   ///< the sinc oversampler
    int shiftFactor;                                                               ///< the oversampling of the shifts of the peaks
};

/// an oversampler of count correlation surfaces of height x width around their peaks, by factor,
/// for the arrays arrayT: FFT or sinc, by method
/// @param rawFactor the oversampling of the windows (the sinc oversampler shifts the peaks by
///     factor*rawFactor)
/// @param stream the stream of the work
template <template <typename> class arrayT>
std::unique_ptr<SurfaceOverSampler<arrayT>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count,
    typename array_traits<arrayT>::stream_type stream)
{
    if (method == OversamplingMethod::Sinc)
        return std::make_unique<SincSurfaceOverSampler<arrayT>>(factor, rawFactor, stream);
    return std::make_unique<FFTSurfaceOverSampler<arrayT>>(height, width, factor, count, stream);
}

} // namespace
//end of file
