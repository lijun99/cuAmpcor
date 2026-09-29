/*
 * @file cuOverSampler.h
 * @brief Oversampling with FFT padding method (CPU backend)
 *
 * Define cuOverSampler class, to save fftw plans and perform oversampling calculations
 * For float images use cuOverSamplerR2R
 * For complex images use cuOverSamplerC2C
 */

#ifndef __CUOVERSAMPLER_H
#define __CUOVERSAMPLER_H

#include "cuArrays.h"
#include "data_types.h"
#include "fftwUtil.h"

namespace pycuampcor::cpu {

// FFT Oversampler for complex images
class cuOverSamplerC2C
{
private:
     fftw_plan_type forwardPlan;   // forward fft handle
     fftw_plan_type backwardPlan;  // backward fft handle
     stream_t stream;              // stream (not used)
     cuArrays<complex_type> *workIn;  // work array to hold forward fft data
     cuArrays<complex_type> *workOut; // work array to hold padded data
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
class cuOverSamplerR2R
{
private:
     fftw_plan_type forwardPlan;
     fftw_plan_type backwardPlan;
     stream_t stream;
     cuArrays<complex_type> *workSizeIn;
     cuArrays<complex_type> *workSizeOut;

public:
    cuOverSamplerR2R() = delete;
    cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_);
    void setStream(stream_t stream_);
    void execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut);
    ~cuOverSamplerR2R();
};

} // namespace

#endif //__CUOVERSAMPLER_H
// end of file
