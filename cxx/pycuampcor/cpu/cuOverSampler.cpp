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
    // work arrays for one image
    workColumns = new cuArrays<complex_type>(outNX, inNY, 1);
    workColumns->allocate();
    workPadded = new cuArrays<complex_type>(outNX, outNY, 1);
    workPadded->allocate();
    workSizeOut = new cuArrays<complex_type>(outNX, outNY, 1);
    workSizeOut->allocate();

    // the numbers of the positive and negative frequency columns, as in cuArraysFFTPaddingMany
    const int nColumns[2] = {(inNY+1)/2, inNY/2};
    // and their positions in workColumns and workPadded
    const int columnIn[2] = {0, nColumns[0]};
    const int columnOut[2] = {0, outNY-nColumns[1]};

    {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        // exp(+i) to the frequency domain and exp(-i) back, as the isce3 (v1) pycuampcor:
        // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
        // forward fft (in-place) of all images
        forwardPlan = planMany2D(inNX, inNY, nImages, workSizeIn->devData, workSizeIn->devData, FFTW_BACKWARD);
        // inverse fft along the columns, from workColumns to the non-zero columns of workPadded
        for(int k = 0; k < 2; k++)
            backwardPlanColumns[k] = (nColumns[k] == 0) ? nullptr : PYCUAMPCOR_FFTW(plan_many_dft)(1, &outNX, nColumns[k],
                fftwCast(workColumns->devData + columnIn[k]), NULL, inNY, 1,
                fftwCast(workPadded->devData + columnOut[k]), NULL, outNY, 1,
                FFTW_FORWARD, PYCUAMPCOR_FFTW_FLAGS | FFTW_PRESERVE_INPUT);
        // inverse fft along the rows, out-of-place to keep the zero columns of workPadded
        backwardPlanRows = PYCUAMPCOR_FFTW(plan_many_dft)(1, &outNY, outNX,
            fftwCast(workPadded->devData), NULL, 1, outNY,
            fftwCast(workSizeOut->devData), NULL, 1, outNY,
            FFTW_FORWARD, PYCUAMPCOR_FFTW_FLAGS | FFTW_PRESERVE_INPUT);
    }

    // the zeros padded in the middle, set once (only the non-zero parts are written in execute)
    workColumns->setZero(stream);
    workPadded->setZero(stream);
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
    // forward fft of all images
    cuArraysCopyPadded(imagesIn, workSizeIn, stream);
    PYCUAMPCOR_FFTW(execute)(forwardPlan);

    const int inNX = workSizeIn->height, inNY = workSizeIn->width;
    const int outNX = workSizeOut->height, outNY = workSizeOut->width;
    const int imagesOutNX = imagesOut->height, imagesOutNY = imagesOut->width;
    // fftw doesn't normalize
    const real_type factor = real_type(1)/workSizeIn->size;
    for(int imageIdx = 0; imageIdx < workSizeIn->count; imageIdx++) {
        // pad zeros in the middle of the columns (as cuArraysFFTPaddingMany), with the spectrum
        // rows moved to the top and bottom; the zero rows are not changed
        const complex_type *spectrum = workSizeIn->devData + (size_t)imageIdx*workSizeIn->size;
        for(int x1 = 0; x1 < inNX; x1++) {
            const int x2 = (x1 < (inNX+1)/2) ? x1 : outNX - inNX + x1;
            for(int y = 0; y < inNY; y++)
                workColumns->devData[IDX2R(x2, y, inNY)] = spectrum[IDX2R(x1, y, inNY)]*factor;
        }
        // inverse fft along the columns, then along the rows
        for(int k = 0; k < 2; k++)
            if(backwardPlanColumns[k])
                PYCUAMPCOR_FFTW(execute)(backwardPlanColumns[k]);
        PYCUAMPCOR_FFTW(execute)(backwardPlanRows);
        // take the real part
        real_type *imageOut = imagesOut->devData + (size_t)imageIdx*imagesOut->size;
        for(int x = 0; x < imagesOutNX; x++)
            for(int y = 0; y < imagesOutNY; y++)
                imageOut[IDX2R(x, y, imagesOutNY)] = workSizeOut->devData[IDX2R(x, y, outNY)].x;
    }
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R()
{
    {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        PYCUAMPCOR_FFTW(destroy_plan)(forwardPlan);
        for(int k = 0; k < 2; k++)
            if(backwardPlanColumns[k])
                PYCUAMPCOR_FFTW(destroy_plan)(backwardPlanColumns[k]);
        PYCUAMPCOR_FFTW(destroy_plan)(backwardPlanRows);
    }
    delete workSizeIn;
    delete workColumns;
    delete workPadded;
    delete workSizeOut;
}

} // namespace
// end of file
