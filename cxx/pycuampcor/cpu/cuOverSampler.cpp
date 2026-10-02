/*
 * @file cuOverSampler.cpp
 * @brief Implementations of cuOverSamplerR2R (C2C) class (CPU backend)
 */

// my declarations
#include "cuOverSampler.h"
// dependencies
#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

// create a batched 2D complex fft plan
static fftw_plan_type planMany2D(int nx, int ny, int nImages, complex_type *in, complex_type *out, int sign)
{
    int n[2] = {nx, ny};
    int size = nx*ny;
    return PYCUAMPCOR_FFTW(plan_many_dft)(2, n, nImages,
        fftwCast(in), NULL, 1, size,
        fftwCast(out), NULL, 1, size,
        sign, PYCUAMPCOR_FFTW_FLAGS);
}

/**
 * Constructor for cuOversamplerC2C
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages batches
 * @param stream_ (not used)
 */
cuOverSamplerC2C::cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    // set up work arrays
    workIn = new cuArrays<complex_type>(inNX, inNY, nImages);
    workIn->allocate();
    workOut = new cuArrays<complex_type>(outNX, outNY, nImages);
    workOut->allocate();

    // set up fft plans (out-of-place), with temporary arrays in place of the input/output images
    cuArrays<complex_type> planIn(inNX, inNY, nImages);
    planIn.allocate();
    cuArrays<complex_type> planOut(outNX, outNY, nImages);
    planOut.allocate();
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    // exp(+i) to the frequency domain and exp(-i) back, as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
    forwardPlan = planMany2D(inNX, inNY, nImages, planIn.devData, workIn->devData, FFTW_BACKWARD);
    backwardPlan = planMany2D(outNX, outNY, nImages, workOut->devData, planOut.devData, FFTW_FORWARD);
}

/**
 * Set up stream (not used)
 */
void cuOverSamplerC2C::setStream(stream_t stream_)
{
    this->stream = stream_;
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void cuOverSamplerC2C::execute(cuArrays<complex_type> *imagesIn, cuArrays<complex_type> *imagesOut)
{
    // FFT to frequency domain
    PYCUAMPCOR_FFTW(execute_dft)(forwardPlan, fftwCast(imagesIn->devData), fftwCast(workIn->devData));
    // padding zeros in the middle
    cuArraysFFTPaddingMany(workIn, workOut, stream);
    // iFFT back to time domain
    PYCUAMPCOR_FFTW(execute_dft)(backwardPlan, fftwCast(workOut->devData), fftwCast(imagesOut->devData));
}

/// destructor
cuOverSamplerC2C::~cuOverSamplerC2C()
{
    // destroy fft handles
    {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        PYCUAMPCOR_FFTW(destroy_plan)(forwardPlan);
        PYCUAMPCOR_FFTW(destroy_plan)(backwardPlan);
    }
    // deallocate work arrays
    delete(workIn);
    delete(workOut);
}

// end of cuOverSamplerC2C

/**
 * Constructor for cuOversamplerR2R
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages the number of images
 * @param stream_ (not used)
 */
cuOverSamplerR2R::cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    workSizeIn = new cuArrays<complex_type>(inNX, inNY, nImages);
    workSizeIn->allocate();
    workSizeOut = new cuArrays<complex_type>(outNX, outNY, nImages);
    workSizeOut->allocate();

    // set up fft plans (in-place)
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    // exp(+i) to the frequency domain and exp(-i) back, as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
    forwardPlan = planMany2D(inNX, inNY, nImages, workSizeIn->devData, workSizeIn->devData, FFTW_BACKWARD);
    backwardPlan = planMany2D(outNX, outNY, nImages, workSizeOut->devData, workSizeOut->devData, FFTW_FORWARD);
}

void cuOverSamplerR2R::setStream(stream_t stream_)
{
    stream = stream_;
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void cuOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut)
{
    cuArraysCopyPadded(imagesIn, workSizeIn, stream);
    PYCUAMPCOR_FFTW(execute)(forwardPlan);
    cuArraysFFTPaddingMany(workSizeIn, workSizeOut, stream);
    PYCUAMPCOR_FFTW(execute)(backwardPlan);
    cuArraysCopyExtract(workSizeOut, imagesOut, make_int2(0,0), stream);
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R()
{
    {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        PYCUAMPCOR_FFTW(destroy_plan)(forwardPlan);
        PYCUAMPCOR_FFTW(destroy_plan)(backwardPlan);
    }
    delete workSizeIn;
    delete workSizeOut;
}

} // namespace
// end of file
