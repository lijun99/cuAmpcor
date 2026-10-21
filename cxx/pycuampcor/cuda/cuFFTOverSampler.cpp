/**
 * @file  cuFFTOverSampler.cpp
 * @brief The FFT oversamplers on the CUDA device (complex and real arrays), and the oversamplers of
 *     the correlation surfaces
 */

// my declarations
#include "cuFFTOverSampler.h"
#include "cuArrays.h"
#include "device.h"
#include "cudaError.h"
#include "float2.h"
#include "cuArraysCopy.h"
#include "cuArraysPadding.h"
#include "logging.h"
#include "cuSincOverSampler.h"
#include "SurfaceOverSampler.h"
#include "traits.h"
#include "Options.h"

namespace pycuampcor {

// the definitions use the CUDA package: its types, kernels and plans
using namespace cuda;

/// the FFT oversampler of complex arrays (windows) on the CUDA device
using FFTOverSamplerC2C = OverSampler<OversamplingMethod::FFT, cuda::complex_type, Device::CUDA>;
/// the FFT oversampler of real arrays (correlation surfaces) on the CUDA device
using FFTOverSamplerR2R = OverSampler<OversamplingMethod::FFT, cuda::real_type, Device::CUDA>;

/**
 * Constructor of the FFT oversampler of complex arrays
 * @param inNX,inNY the size of the input images
 * @param outNX,outNY the size of the output (oversampled) images
 * @param nImages the number of images
 * @param stream_ the CUDA stream of the transforms
 */
FFTOverSamplerC2C::OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_)
    : stream(stream_)
{

    int inNXp2 = inNX;
    int inNYp2 = inNY;
    int outNXp2 = outNX;
    int outNYp2 = outNY;

    // set up work arrays
    workIn = cuArrays<complex_type>(inNXp2, inNYp2, nImages);
    workOut = cuArrays<complex_type>(outNXp2, outNYp2, nImages);

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
    // the transforms are queued on the stream
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void FFTOverSamplerC2C::execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut)
{
    // the input of the (out-of-place) forward FFT is not written (cufft takes it as non-const)
    complex_type *input = const_cast<complex_type *>(imagesIn.data_handle());
    // FFT to frequency domain, with exp(+i) and back with exp(-i) as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
 #ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, input, workIn.data_handle(), CUFFT_INVERSE));
 #else
    checkCudaErrors(cufftExecC2C(forwardPlan, input, workIn.data_handle(), CUFFT_INVERSE));
 #endif

    // padding zeros in the middle
    cuArraysFFTPaddingMany(workIn, workOut, stream);

    // iFFT back to time domain
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workOut.data_handle(), imagesOut.data_handle(), CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workOut.data_handle(), imagesOut.data_handle(), CUFFT_FORWARD));
#endif
}

/**
 * Release the cufft plans and the work arrays
 */
FFTOverSamplerC2C::~OverSampler() = default;


/**
 * Constructor of the FFT oversampler of real arrays
 * @param inNX,inNY the size of the input images
 * @param outNX,outNY the size of the output (oversampled) images
 * @param nImages the number of images
 * @param stream_ the CUDA stream of the transforms
 */
FFTOverSamplerR2R::OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, cudaStream_t stream_)
    : stream(stream_)
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
    workSizeIn = cuArrays<complex_type>(inNXp2, inNYp2, nImages);
    workSizeOut = cuArrays<complex_type>(outNXp2, outNYp2, nImages);
#ifdef CUAMPCOR_DOUBLE
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_Z2Z, nImages);
    backwardPlan.planMany(NRANK, nUpSample, nullptr, 1, fImageUpSampleSize, nullptr, 1, outNX*outNY, CUFFT_Z2Z, nImages);
#else
    forwardPlan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, fImageSize, CUFFT_C2C, nImages);
    backwardPlan.planMany(NRANK, nUpSample, nullptr, 1, fImageUpSampleSize, nullptr, 1, outNX*outNY, CUFFT_C2C, nImages);
#endif
    // the transforms are queued on the stream
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void FFTOverSamplerR2R::execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut)
{

    cuArraysCopyPadded(imagesIn, workSizeIn, stream);

    // exp(+i) to the frequency domain and exp(-i) back, as the oversampler of complex arrays
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(forwardPlan, workSizeIn.data_handle(), workSizeIn.data_handle(), CUFFT_INVERSE));
#else
    checkCudaErrors(cufftExecC2C(forwardPlan, workSizeIn.data_handle(), workSizeIn.data_handle(), CUFFT_INVERSE));
#endif

    cuArraysFFTPaddingMany(workSizeIn, workSizeOut, stream);

#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(backwardPlan, workSizeOut.data_handle(), workSizeOut.data_handle(), CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(backwardPlan, workSizeOut.data_handle(), workSizeOut.data_handle(), CUFFT_FORWARD));
#endif

    cuArraysCopyExtract(workSizeOut, imagesOut, make_int2(0,0), stream);
}

/**
 * Release the cufft plans and the work arrays
 */
FFTOverSamplerR2R::~OverSampler() = default;

} // namespace

namespace pycuampcor::cuda {

namespace {

/// the FFT oversampler of correlation surfaces (the whole surfaces)
class FFTSurfaceOverSampler : public SurfaceOverSampler<cuArrays> {
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
    OverSampler<OversamplingMethod::FFT, real_type, Device::CUDA> oversampler;   ///< the FFT oversampler
};

/// the sinc oversampler of correlation surfaces (around their peaks only)
class SincSurfaceOverSampler : public SurfaceOverSampler<cuArrays> {
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
    OverSampler<OversamplingMethod::Sinc, real_type, Device::CUDA> oversampler;   ///< the sinc oversampler
    int shiftFactor;                    ///< the oversampling of the shifts of the peaks
};

} // namespace

/**
 * An oversampler of correlation surfaces around their peaks: FFT or sinc, by method
 */
std::unique_ptr<SurfaceOverSampler<cuArrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, cudaStream_t stream)
{
    if (method == OversamplingMethod::Sinc)
        return std::make_unique<SincSurfaceOverSampler>(factor, rawFactor, stream);
    return std::make_unique<FFTSurfaceOverSampler>(height, width, factor, count, stream);
}

} // namespace