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

#include "cuArrays.h"
#include "data_types.h"
#include "cufftPlan.h"

#include <memory>

namespace pycuampcor::cuda {

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     CufftPlan forwardPlan;   // forward fft handle
     CufftPlan backwardPlan;  // backward fft handle
     cudaStream_t stream;       // cuda stream
     std::unique_ptr<cuArrays<complex_type>> workIn;  // work array to hold forward fft data
     std::unique_ptr<cuArrays<complex_type>> workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
     // set cuda stream
     void setStream(cudaStream_t stream_);
     // execute oversampling
     void execute(cuArrays<complex_type> *imagesIn, cuArrays<complex_type> *imagesOut);
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
     std::unique_ptr<cuArrays<complex_type>> workSizeIn;
     std::unique_ptr<cuArrays<complex_type>> workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_);
    void setStream(cudaStream_t stream_);
    void execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut);
    ~cuOverSamplerR2R();
};

} // namespace
// end of file

