/*
 * @file cuOverSampler.cu
 * @brief Implementations of cuOverSamplerR2R (C2C) class
 */

// my declarations
#include "cuOverSampler.h"
#include "cuArrays.h"
#include "device.h"
#include "cudaError.h"
#include "float2.h"
#include "cuArraysCopy.h"
#include "cuArraysPadding.h"
#include "logging.h"
#include "cuSincOverSampler.h"
#include "cuSurfaceOverSampler.h"
#include "arrayTraits.h"
#include "cuAmpcorOptions.h"

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
void cuOverSamplerC2C::execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut)
{
    // the input of the (out-of-place) forward FFT is not written (cufft takes it as non-const)
    complex_type *input = const_cast<complex_type *>(imagesIn.data_handle());
    // FFT to frequency domain, with exp(+i) and back with exp(-i) as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
 #ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, input, workIn->devData, CUFFT_INVERSE));
 #else
    checkCudaErrors(cufftExecC2C(forwardPlan, input, workIn->devData, CUFFT_INVERSE));
 #endif

    // padding zeros in the middle
    cuArraysFFTPaddingMany(*workIn, *workOut, stream);

    // iFFT back to time domain
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workOut->devData, imagesOut.data_handle(), CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workOut->devData, imagesOut.data_handle(), CUFFT_FORWARD));
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
void cuOverSamplerR2R::execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut)
{

    cuArraysCopyPadded(imagesIn, *workSizeIn, stream);

    // exp(+i) to the frequency domain and exp(-i) back, as cuOverSamplerC2C
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
#else
    checkCudaErrors(cufftExecC2C(forwardPlan, workSizeIn->devData, workSizeIn->devData, CUFFT_INVERSE));
#endif

    cuArraysFFTPaddingMany(*workSizeIn, *workSizeOut, stream);

#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workSizeOut->devData, workSizeOut->devData, CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workSizeOut->devData, workSizeOut->devData, CUFFT_FORWARD));
#endif

    cuArraysCopyExtract(*workSizeOut, imagesOut, make_int2(0,0), stream);
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R() = default;

// end of file

namespace {

/// the FFT oversampler of correlation surfaces (the whole surfaces)
class FFTSurfaceOverSampler : public cuSurfaceOverSampler<cuArrays> {
public:
    /// an oversampler of count surfaces of height x width, by factor
    FFTSurfaceOverSampler(int height, int width, int factor, int count, cudaStream_t stream)
        : oversampler(height, width, height*factor, width*factor, count, stream) {}

    /// oversample the surfaces (the peaks need no shift: the whole surfaces are oversampled)
    void execute(ArraysView<const real_type> surfaces, ArraysView<real_type> oversampled,
        ArraysView<const int2_type>) override
    {
        oversampler.execute(surfaces, oversampled);
    }

private:
    cuOverSamplerR2R oversampler;   ///< the FFT oversampler
};

/// the sinc oversampler of correlation surfaces (around their peaks only)
class SincSurfaceOverSampler : public cuSurfaceOverSampler<cuArrays> {
public:
    /// an oversampler by factor; the peaks are shifted by factor*rawFactor
    SincSurfaceOverSampler(int factor, int rawFactor, cudaStream_t stream)
        : oversampler(factor, stream), shiftFactor(factor*rawFactor) {}

    /// oversample the surfaces around their peaks, shifted from the centers by shift (or not, if empty)
    void execute(ArraysView<const real_type> surfaces, ArraysView<real_type> oversampled,
        ArraysView<const int2_type> shift) override
    {
        if (!shift.empty())
            oversampler.execute(surfaces, oversampled, shift, shiftFactor);
        else
            oversampler.execute(surfaces, oversampled, make_int2(0, 0), shiftFactor);
    }

private:
    cuSincOverSamplerR2R oversampler;   ///< the sinc oversampler
    int shiftFactor;                    ///< the oversampling of the shifts of the peaks
};

} // namespace

/**
 * An oversampler of correlation surfaces around their peaks: FFT or sinc, by method
 */
std::unique_ptr<cuSurfaceOverSampler<cuArrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, cudaStream_t stream)
{
    if (method == OversamplingMethod::Sinc)
        return std::make_unique<SincSurfaceOverSampler>(factor, rawFactor, stream);
    return std::make_unique<FFTSurfaceOverSampler>(height, width, factor, count, stream);
}

} // namespace