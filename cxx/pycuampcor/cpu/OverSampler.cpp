/*
 * @file OverSampler.cpp
 * @brief Implementations of cuOverSamplerR2R (C2C) class (CPU backend)
 */

// my declarations
#include "OverSampler.h"
#include "float2.h"
#include "ArraysCopy.h"
#include "ArraysPadding.h"
#include "indexing.h"
#include "Arrays.h"
#include "SincOverSampler.h"
#include "cuSurfaceOverSampler.h"
#include "arrayTraits.h"
#include "cuAmpcorOptions.h"
// dependencies

namespace pycuampcor::cpu {

// create a batched 2D complex fft plan
static fftw_plan_type planMany2D(int nx, int ny, int nImages, complex_type *in, complex_type *out, int sign)
{
    int n[2] = {nx, ny};
    int size = nx*ny;
    return PYCUAMPCOR_FFTW(plan_many_dft)(2, n, nImages,
        fftwCast(in), nullptr, 1, size,
        fftwCast(out), nullptr, 1, size,
        sign, fftwFlags);
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
    workIn = std::make_unique<array_type<complex_type>>(inNX, inNY, nImages);
    workIn->allocate();
    workOut = std::make_unique<array_type<complex_type>>(outNX, outNY, nImages);
    workOut->allocate();

    // set up fft plans (out-of-place), with temporary arrays in place of the input/output images
    array_type<complex_type> planIn(inNX, inNY, nImages);
    planIn.allocate();
    array_type<complex_type> planOut(outNX, outNY, nImages);
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
void cuOverSamplerC2C::execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut)
{
    // FFT to frequency domain (the input of the out-of-place transform is not written,
    // fftw takes it as non-const)
    PYCUAMPCOR_FFTW(execute_dft)(forwardPlan, fftwCast(const_cast<complex_type *>(imagesIn.data_handle())),
        fftwCast(workIn->devData));
    // padding zeros in the middle
    cuArraysFFTPaddingMany(*workIn, *workOut, stream);
    // iFFT back to time domain
    PYCUAMPCOR_FFTW(execute_dft)(backwardPlan, fftwCast(workOut->devData), fftwCast(imagesOut.data_handle()));
}

/// destructor
cuOverSamplerC2C::~cuOverSamplerC2C() = default;

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
    workSizeIn = std::make_unique<array_type<complex_type>>(inNX, inNY, nImages);
    workSizeIn->allocate();
    // work arrays for one image
    workColumns = std::make_unique<array_type<complex_type>>(outNX, inNY, 1);
    workColumns->allocate();
    workPadded = std::make_unique<array_type<complex_type>>(outNX, outNY, 1);
    workPadded->allocate();
    workSizeOut = std::make_unique<array_type<complex_type>>(outNX, outNY, 1);
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
                fftwCast(workColumns->devData + columnIn[k]), nullptr, inNY, 1,
                fftwCast(workPadded->devData + columnOut[k]), nullptr, outNY, 1,
                FFTW_FORWARD, fftwFlags | FFTW_PRESERVE_INPUT);
        // inverse fft along the rows, out-of-place to keep the zero columns of workPadded
        backwardPlanRows = PYCUAMPCOR_FFTW(plan_many_dft)(1, &outNY, outNX,
            fftwCast(workPadded->devData), nullptr, 1, outNY,
            fftwCast(workSizeOut->devData), nullptr, 1, outNY,
            FFTW_FORWARD, fftwFlags | FFTW_PRESERVE_INPUT);
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
void cuOverSamplerR2R::execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut)
{
    // forward fft of all images
    cuArraysCopyPadded(imagesIn, *workSizeIn, stream);
    PYCUAMPCOR_FFTW(execute)(forwardPlan);

    const int inNX = workSizeIn->height, inNY = workSizeIn->width;
    const int outNX = workSizeOut->height, outNY = workSizeOut->width;
    const int imagesOutNX = imagesOut.height(), imagesOutNY = imagesOut.width();
    // fftw doesn't normalize
    const real_type factor = real_type(1)/workSizeIn->size;
    for(int imageIdx = 0; imageIdx < workSizeIn->count; imageIdx++) {
        // pad zeros in the middle of the columns (as cuArraysFFTPaddingMany), with the spectrum
        // rows moved to the top and bottom; the zero rows are not changed
        const complex_type *spectrum = workSizeIn->devData + static_cast<size_t>(imageIdx)*workSizeIn->size;
        for(int x1 = 0; x1 < inNX; x1++) {
            const int x2 = (x1 < (inNX+1)/2) ? x1 : outNX - inNX + x1;
            for(int y = 0; y < inNY; y++)
                workColumns->devData[idx2r(x2, y, inNY)] = spectrum[idx2r(x1, y, inNY)]*factor;
        }
        // inverse fft along the columns, then along the rows
        for(int k = 0; k < 2; k++)
            if(backwardPlanColumns[k])
                PYCUAMPCOR_FFTW(execute)(backwardPlanColumns[k]);
        PYCUAMPCOR_FFTW(execute)(backwardPlanRows);
        // take the real part
        real_type *imageOut = imagesOut.array(imageIdx).data_handle();
        for(int x = 0; x < imagesOutNX; x++)
            for(int y = 0; y < imagesOutNY; y++)
                imageOut[idx2r(x, y, imagesOutNY)] = workSizeOut->devData[idx2r(x, y, outNY)].x;
    }
}

/// destructor
cuOverSamplerR2R::~cuOverSamplerR2R() = default;

namespace {

/// the FFT oversampler of correlation surfaces (the whole surfaces)
class FFTSurfaceOverSampler : public cuSurfaceOverSampler<Arrays> {
public:
    /// an oversampler of count surfaces of height x width, by factor
    FFTSurfaceOverSampler(int height, int width, int factor, int count, stream_t stream)
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
class SincSurfaceOverSampler : public cuSurfaceOverSampler<Arrays> {
public:
    /// an oversampler by factor; the peaks are shifted by factor*rawFactor
    SincSurfaceOverSampler(int factor, int rawFactor, stream_t stream)
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
std::unique_ptr<cuSurfaceOverSampler<Arrays>> newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, stream_t stream)
{
    if (method == OversamplingMethod::Sinc)
        return std::make_unique<SincSurfaceOverSampler>(factor, rawFactor, stream);
    return std::make_unique<FFTSurfaceOverSampler>(height, width, factor, count, stream);
}

} // namespace
// end of file
