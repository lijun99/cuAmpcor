/**
 * @file  OverSampler.h
 * @brief Oversampling with FFT padding method (CPU backend)
 *
 * Define cuOverSampler class, to save fftw plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Arrays.h"
#include "fftwUtil.h"

namespace pycuampcor::cpu {

/// the FFT oversampler for complex images (windows), by zero padding their spectra
class cuOverSamplerC2C
{
private:
     FftwPlan forwardPlan;               ///< the forward fft handle
     FftwPlan backwardPlan;              ///< the backward fft handle
     stream_t stream;                    ///< the stream (not used)
     array_type<complex_type> workIn;    ///< the work array to hold the forward fft data
     array_type<complex_type> workOut;   ///< the work array to hold the padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_);
     /// execute the oversampling (imagesOut of outNX x outNY)
     void execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut);
     /// destructor
     ~cuOverSamplerC2C();
};

/// the FFT oversampler for real images (correlation surfaces), by zero padding their spectra;
/// the inverse fft of the padded spectrum is done for one image at a time, along the columns
/// (only those with non-zero spectrum) and then along the rows
class cuOverSamplerR2R
{
private:
     FftwPlan forwardPlan;                   ///< the forward fft (in place) of all images
     FftwPlan backwardPlanColumns[2];        ///< the inverse fft along the positive/negative frequency columns (or null)
     FftwPlan backwardPlanRows;              ///< the inverse fft along the rows
     stream_t stream;                        ///< the stream (not used)
     array_type<complex_type> workSizeIn;    ///< the spectra of the input images
     array_type<complex_type> workColumns;   ///< the spectrum of one image, padded along the columns
     array_type<complex_type> workPadded;    ///< the inverse fft along the columns, padded along the rows
     array_type<complex_type> workSizeOut;   ///< the oversampled image

public:
    cuOverSamplerR2R() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_);
    /// execute the oversampling (imagesOut takes the top left part of the oversampled images)
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut);
    /// destructor
    ~cuOverSamplerR2R();
};

/// an oversampler of count correlation surfaces of height x width around their peaks, by factor:
/// FFT (cuOverSamplerR2R) or sinc (cuSincOverSamplerR2R), by method; with rawFactor, the
/// oversampling of the windows (the sinc oversampler shifts the peaks by factor*rawFactor)
std::unique_ptr<SurfaceOverSampler<Arrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, stream_t stream);

} // namespace
// end of file
