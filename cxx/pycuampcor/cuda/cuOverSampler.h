/**
 * @file  cuOverSampler.h
 * @brief Oversampling with FFT padding method
 *
 * Define cuOverSampler class, to save cufft plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 * @todo use template class to unify these two classes
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "cuArrays.h"
#include "cufftPlan.h"

namespace pycuampcor::cuda {

/// the FFT oversampler for complex images (windows), by zero padding their spectra
class cuOverSamplerC2C
{
private:
     CufftPlan forwardPlan;              ///< the forward fft handle
     CufftPlan backwardPlan;             ///< the backward fft handle
     cudaStream_t stream;                ///< the cuda stream
     cuArrays<complex_type> workIn;      ///< the work array to hold the forward fft data
     cuArrays<complex_type> workOut;     ///< the work array to hold the padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
     /// execute the oversampling (imagesOut of outNX x outNY)
     void execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut);
     /// destructor
     ~cuOverSamplerC2C();
};

/// the FFT oversampler for real images (correlation surfaces), by zero padding their spectra
class cuOverSamplerR2R
{
private:
     CufftPlan forwardPlan;                  ///< the forward fft (in place) of all images
     CufftPlan backwardPlan;                 ///< the inverse fft (in place) of the padded spectra
     cudaStream_t stream;                    ///< the cuda stream
     cuArrays<complex_type> workSizeIn;      ///< the input images (as complex), then their spectra
     cuArrays<complex_type> workSizeOut;     ///< the padded spectra, then the oversampled images

public:
    cuOverSamplerR2R() = delete;
    /// constructor, for nImages images of inNX x inNY oversampled to outNX x outNY
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    /// execute the oversampling (imagesOut takes the top left part of the oversampled images)
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut);
    /// destructor
    ~cuOverSamplerR2R();
};

/// an oversampler of count correlation surfaces of height x width around their peaks, by factor:
/// FFT (cuOverSamplerR2R) or sinc (cuSincOverSamplerR2R), by method; with rawFactor, the
/// oversampling of the windows (the sinc oversampler shifts the peaks by factor*rawFactor)
std::unique_ptr<SurfaceOverSampler<cuArrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, cudaStream_t stream);

} // namespace
// end of file

