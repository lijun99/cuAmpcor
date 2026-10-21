/**
 * @file  OverSampler.h
 * @brief The oversamplers of batches of 2D arrays: one class template, by method, data type and device
 *
 * OverSampler<method, T, device> oversamples count arrays of T (windows, or correlation surfaces)
 * with the method (FFT, by zero padding the spectra, or sinc interpolation) on the device of the
 * arrays (Device::CPU, Device::CUDA). There is no generic definition: the packages define the
 * specializations they implement,
 * - OverSampler<OversamplingMethod::FFT, complex_type, device>: complex arrays (the windows), by
 *   zero padding their spectra (FFTOverSampler.h, cuFFTOverSampler.h);
 * - OverSampler<OversamplingMethod::FFT, real_type, device>: real arrays (the correlation
 *   surfaces), by zero padding their spectra (same files);
 * - OverSampler<OversamplingMethod::Sinc, real_type, device>: real arrays (the correlation
 *   surfaces), by sinc interpolation within a window around their peaks (SincOverSampler.h,
 *   cuSincOverSampler.h).
 * The processors take the device from the arrays: OverSampler<method, T, array_traits<arrayT>::device>.
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"
#include "Options.h"

namespace pycuampcor {

/// an oversampler of count 2D arrays of T, by method, on device (declared only: the packages
/// define the specializations they implement)
template <OversamplingMethod method, typename T, Device device>
class OverSampler;

} // namespace
//end of file
