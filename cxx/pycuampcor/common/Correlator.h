/**
 * @file  Correlator.h
 * @brief The cross-correlators of batches of windows: one class template, by algorithm, data type
 *     and device
 *
 * Correlator<algorithm, T, device> cross-correlates count reference windows (templates) with
 * count secondary windows (images) of T, giving count correlation surfaces of (secondary -
 * reference + 1) pixels each way, with the algorithm on the device of the arrays (Device::CPU,
 * Device::CUDA). There is no generic definition: the packages define the specializations they
 * implement, all with the constructor (height, width, count, stream) for count secondary windows
 * of height x width, and execute(templates, images, results):
 * - Correlator<Algorithm::Frequency, real_type, device>: the amplitudes of the windows, in the
 *   frequency domain (CorrFrequency.h, cuCorrFrequency.h);
 * - Correlator<Algorithm::Time, real_type, device>: the amplitudes of the windows, in the time
 *   domain (CorrTimeDomain.h, cuCorrTimeDomain.h);
 * - Correlator<Algorithm::Coherence, complex_type, device>: the complex windows, in the frequency
 *   domain, normalized to the squared coherence (CorrCoherence.h, cuCorrCoherence.h); used directly by
 *   the two-pass workflow (on complex windows), not through WindowCorrelator.
 *
 * The processors choose the algorithm at run time (Parameter::algorithm): WindowCorrelator is the
 * interface, implemented by WindowCorrelatorBy for each algorithm on the correlators of the
 * device of the arrays, and newWindowCorrelator creates one by the algorithm. The implementations
 * are instantiated with the processors (the packages' processors.cpp), where the correlators of
 * the package are defined.
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"
#include "ArraysView.h"
#include "Options.h"

#include <memory>
#include <stdexcept>

namespace pycuampcor {

/// a cross-correlator of count windows of T, by algorithm, on device (declared only: the packages
/// define the specializations they implement)
template <Algorithm algorithm, typename T, Device device>
class Correlator;

/// the cross-correlation of the amplitudes of the windows, by an algorithm chosen at run time,
/// for the arrays arrayT
template <template <typename> class arrayT>
class WindowCorrelator {
public:
    using real_type = typename array_traits<arrayT>::real_type;  ///< the real type

    virtual ~WindowCorrelator() = default;

    /// compute the correlation surfaces (results) of the reference windows (templates) with the
    /// secondary windows (images)
    virtual void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results) = 0;
};

/// the cross-correlation of the amplitudes of the windows by the algorithm, for the arrays arrayT:
/// the correlator Correlator<algorithm, real_type, device> of the device of the arrays
template <template <typename> class arrayT, Algorithm algorithm>
class WindowCorrelatorBy : public WindowCorrelator<arrayT> {
public:
    using traits = array_traits<arrayT>;                   ///< the types that go with the arrays
    using real_type = typename traits::real_type;          ///< the real type
    using stream_type = typename traits::stream_type;      ///< the stream of the work

    /// a correlator for count secondary windows of height x width
    WindowCorrelatorBy(int height, int width, int count, stream_type stream)
        : correlator(height, width, count, stream) {}

    /// compute the correlation surfaces (see WindowCorrelator::execute)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results) override
    {
        correlator.execute(templates, images, results);
    }

private:
    Correlator<algorithm, real_type, traits::device> correlator;   ///< the correlator
};

/// a cross-correlator of the amplitudes of count windows, by the algorithm, for the arrays arrayT
/// @param height,width the size of the secondary windows
/// @param stream the stream of the work
template <template <typename> class arrayT>
std::unique_ptr<WindowCorrelator<arrayT>> newWindowCorrelator(Algorithm algorithm,
    int height, int width, int count, typename array_traits<arrayT>::stream_type stream)
{
    switch (algorithm) {
        case Algorithm::Frequency:
            return std::make_unique<WindowCorrelatorBy<arrayT, Algorithm::Frequency>>(height, width, count, stream);
        case Algorithm::Time:
            return std::make_unique<WindowCorrelatorBy<arrayT, Algorithm::Time>>(height, width, count, stream);
        case Algorithm::Coherence:
            throw std::invalid_argument("The coherence is correlated by the complex windows, not their amplitudes");
        default:
            throw std::invalid_argument("Unsupported correlation algorithm");
    }
}

} // namespace
//end of file
