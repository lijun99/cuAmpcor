/*
 * @file cuOverSampler.cu
 * @brief Implementations of cuOverSamplerR2R (C2C) class
 */

// my declarations
#include "public.h"

// dependencies

namespace pycuampcor::cuda {

/**
 * Constructor for cuOversamplerC2C
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages batches
 * @param stream_ cuda stream
 */
cuOverSamplerC2C::cuOverSamplerC2C(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_)
{

    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY;
    */

    // set up work arrays
    workIn = std::make_unique<cuArrays<complex_type>>(inNXp2, inNYp2, nImages);
    workIn->allocate();
    workOut = std::make_unique<cuArrays<complex_type>>(outNXp2, outNYp2, nImages);
    workOut->allocate();

    // set up fft plans
    int imageSize = inNXp2*inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nOverSample[NRANK] = {outNXp2, outNYp2};
    int fImageOverSampleSize = outNXp2*outNYp2;
#ifdef CUAMPCOR_DOUBLE
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_Z2Z, nImages);
    backwardPlan.planMany(NRANK, nOverSample, nullptr, 1, fImageOverSampleSize, nullptr, 1, fImageOverSampleSize, CUFFT_Z2Z, nImages);
#else
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_C2C, nImages);
    backwardPlan.planMany(NRANK, nOverSample, nullptr, 1, fImageOverSampleSize, nullptr, 1, fImageOverSampleSize, CUFFT_C2C, nImages);
#endif
    // set cuda stream
    setStream(stream_);
}

/**
 * Set up cuda stream
 */
void cuOverSamplerC2C::setStream(cudaStream_t stream_)
{
    this->stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 * @param[in] method phase deramping method
 * @param[in] axis deramping axis
 */
void cuOverSamplerC2C::execute(cuArrays<complex_type> *imagesIn, cuArrays<complex_type> *imagesOut)
{
    // FFT to frequency domain, with exp(+i) and back with exp(-i) as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
 #ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, imagesIn->devData, workIn->devData, CUFFT_INVERSE));
 #else
    checkCudaErrors(cufftExecC2C(forwardPlan, imagesIn->devData, workIn->devData, CUFFT_INVERSE));
 #endif

    // padding zeros in the middle
    cuArraysFFTPaddingMany(workIn.get(), workOut.get(), stream);

    // iFFT back to time domain
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workOut->devData, imagesOut->devData, CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workOut->devData, imagesOut->devData, CUFFT_FORWARD));
#endif
}

/// destructor
cuOverSamplerC2C::~cuOverSamplerC2C() = default;

// end of cuOverSamplerC2C

/**
 * Constructor for cuOversamplerR2R
 * @param input image size inNX x inNY
 * @param output image size outNX x outNY
 * @param nImages the number of images
 * @param stream_ cuda stream
 */
cuOverSamplerR2R::cuOverSamplerR2R(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream)
{

    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

    /* if expanded to 2^n
    int inNXp2 = nextpower2(inNX);
    int inNYp2 = nextpower2(inNY);
    int outNXp2 = inNXp2*outNX/inNX;
    int outNYp2 = inNYp2*outNY/inNY;
    */

    int imageSize = inNXp2 *inNYp2;
    int n[NRANK] ={inNXp2, inNYp2};
    int fImageSize = inNXp2*inNYp2;
    int nUpSample[NRANK] = {outNXp2, outNYp2};
    int fImageUpSampleSize = outNXp2*outNYp2;
    workSizeIn = std::make_unique<cuArrays<complex_type>>(inNXp2, inNYp2, nImages);
    workSizeIn->allocate();
    workSizeOut = std::make_unique<cuArrays<complex_type>>(outNXp2, outNYp2, nImages);
    workSizeOut->allocate();
#ifdef CUAMPCOR_DOUBLE
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_Z2Z, nImages);
    backwardPlan.planMany(NRANK, nUpSample, nullptr, 1, fImageUpSampleSize, nullptr, 1, outNX*outNY, CUFFT_Z2Z, nImages);
#else
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_C2C, nImages);
    backwardPlan.planMany(NRANK, nUpSample, nullptr, 1, fImageUpSampleSize, nullptr, 1, outNX*outNY, CUFFT_C2C, nImages);
#endif
    setStream(stream);
}

void cuOverSamplerR2R::setStream(cudaStream_t stream_)
{
    stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void cuOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut)
{

    cuArraysCopyPadded(imagesIn, workSizeIn.get(), stream);

    // exp(+i) to the frequency domain and exp(-i) back, as cuOverSamplerC2C
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
#else
    checkCudaErrors(cufftExecC2C(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
#endif

    cuArraysFFTPaddingMany(workSizeIn.get(), workSizeOut.get(), stream);

#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workSizeOut->devData, workSizeOut->devData, CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workSizeOut->devData, workSizeOut->devData, CUFFT_FORWARD));
#endif

    cuArraysCopyExtract(workSizeOut.get(), imagesOut, make_int2(0,0), stream);
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R() = default;

// end of file

} // namespace