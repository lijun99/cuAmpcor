/*
 * @file cuOverSampler.h
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

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     CufftPlan forwardPlan;   // forward fft handle
     CufftPlan backwardPlan;  // backward fft handle
     cudaStream_t stream;       // cuda stream
     cuArrays<complex_type> workIn;  // work array to hold forward fft data
     cuArrays<complex_type> workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
     // set cuda stream
     void setStream(cudaStream_t stream_);
     // execute oversampling
     void execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut);
     // destructor
     ~cuOverSamplerC2C();
};

// FFT Oversampler for complex images
class cuOverSamplerR2R
{
private:
     CufftPlan forwardPlan;
     CufftPlan backwardPlan;
     cudaStream_t stream;
     cuArrays<complex_type> workSizeIn;
     cuArrays<complex_type> workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    void setStream(cudaStream_t stream_);
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut);
    ~cuOverSamplerR2R();
};

/// an oversampler of count correlation surfaces of height x width around their peaks, by factor:
/// FFT (cuOverSamplerR2R) or sinc (cuSincOverSamplerR2R), by method; with rawFactor, the
/// oversampling of the windows (the sinc oversampler shifts the peaks by factor*rawFactor)
std::unique_ptr<cuSurfaceOverSampler<cuArrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, cudaStream_t stream);

} // namespace
// end of file

