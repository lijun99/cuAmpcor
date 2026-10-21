/*
 * @file cuOverSampler.h
 * @brief Oversampling with FFT padding method (CPU backend)
 *
 * Define cuOverSampler class, to save fftw plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 */

#pragma once

#include "forward.h"
#include "fftwUtil.h"

namespace pycuampcor::cpu {

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     FftwPlan forwardPlan;   // forward fft handle
     FftwPlan backwardPlan;  // backward fft handle
     stream_t stream;              // stream (not used)
     std::unique_ptr<cuArrays<complex_type>> workIn;  // work array to hold forward fft data
     std::unique_ptr<cuArrays<complex_type>> workOut; // work array to hold padded data
public:
     // disable the default constructor
     cuOverSamplerC2C() = delete;
     // constructor
     cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_);
     // set stream
     void setStream(stream_t stream_);
     // execute oversampling
     void execute(cuArrays<complex_type> *imagesIn, cuArrays<complex_type> *imagesOut);
     // destructor
     ~cuOverSamplerC2C();
};

// FFT Oversampler for real images
// the inverse fft of the padded spectrum is done for one image at a time, along the columns
// (only those with non-zero spectrum) and then along the rows
class cuOverSamplerR2R
{
private:
     FftwPlan forwardPlan;
     FftwPlan backwardPlanColumns[2]; // for the positive and negative frequency columns
     FftwPlan backwardPlanRows;
     stream_t stream;
     std::unique_ptr<cuArrays<complex_type>> workSizeIn;   // the spectra of the input images
     std::unique_ptr<cuArrays<complex_type>> workColumns;  // the spectrum of one image, padded along the columns
     std::unique_ptr<cuArrays<complex_type>> workPadded;   // the inverse fft along the columns, padded along the rows
     std::unique_ptr<cuArrays<complex_type>> workSizeOut;  // the oversampled image

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_);
    void setStream(stream_t stream_);
    void execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut);
    ~cuOverSamplerR2R();
};

} // namespace
// end of file
