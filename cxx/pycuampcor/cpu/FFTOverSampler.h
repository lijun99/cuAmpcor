/**
 * @file  FFTOverSampler.h
 * @brief The FFT oversamplers on the CPU: OverSampler<OversamplingMethod::FFT, T, Device::CPU> for
 *     complex arrays (windows) and real arrays (correlation surfaces)
 *
 * The arrays are oversampled by zero padding their spectra (fftw plans saved for the batches).
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "OverSampler.h"
#include "Arrays.h"
#include "fftwUtil.h"

namespace pycuampcor {

/// the FFT oversampler of complex arrays (windows) on the CPU, by zero padding their spectra
template <>
class OverSampler<OversamplingMethod::FFT, cpu::complex_type, Device::CPU>
{
public:
    using complex_type = cpu::complex_type;   ///< the complex type

    // disable the default constructor
    OverSampler() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cpu::stream_t stream_);
    /// execute the oversampling (imagesOut of outNX x outNY)
    void execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut);
    /// destructor
    ~OverSampler();

private:
    cpu::FftwPlan forwardPlan;               ///< the forward fft handle
    cpu::FftwPlan backwardPlan;              ///< the backward fft handle
    cpu::stream_t stream;                    ///< the stream (not used)
    cpu::Arrays<complex_type> workIn;        ///< the work array to hold the forward fft data
    cpu::Arrays<complex_type> workOut;       ///< the work array to hold the padded data
};

/// the FFT oversampler of real arrays (correlation surfaces) on the CPU, by zero padding their
/// spectra; the inverse fft of the padded spectrum is done for one image at a time, along the
/// columns (only those with non-zero spectrum) and then along the rows
template <>
class OverSampler<OversamplingMethod::FFT, cpu::real_type, Device::CPU>
{
public:
    using real_type = cpu::real_type;         ///< the real type
    using complex_type = cpu::complex_type;   ///< the complex type (of the spectra)

    OverSampler() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cpu::stream_t stream_);
    /// execute the oversampling (imagesOut takes the top left part of the oversampled images)
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut);
    /// destructor
    ~OverSampler();

private:
    cpu::FftwPlan forwardPlan;                ///< the forward fft (in place) of all images
    cpu::FftwPlan backwardPlanColumns[2];     ///< the inverse fft along the positive/negative frequency columns (or null)
    cpu::FftwPlan backwardPlanRows;           ///< the inverse fft along the rows
    cpu::stream_t stream;                     ///< the stream (not used)
    cpu::Arrays<complex_type> workSizeIn;     ///< the spectra of the input images
    cpu::Arrays<complex_type> workColumns;    ///< the spectrum of one image, padded along the columns
    cpu::Arrays<complex_type> workPadded;     ///< the inverse fft along the columns, padded along the rows
    cpu::Arrays<complex_type> workSizeOut;    ///< the oversampled image
};

} // namespace
// end of file
