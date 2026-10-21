/**
 * @file  FFTOverSampler.cpp
 * @brief The FFT oversamplers on the CPU (complex and real arrays)
 */

// my declarations
#include "FFTOverSampler.h"
#include "float2.h"
#include "ArraysCopy.h"
#include "ArraysPadding.h"
#include "Arrays.h"
#include "Options.h"

#include <utility>

namespace pycuampcor {

// the definitions use the CPU package: its types, arrays and fftw plans
using namespace cpu;

/// the FFT oversampler of complex arrays (windows) on the CPU
using FFTOverSamplerC2C = OverSampler<OversamplingMethod::FFT, cpu::complex_type, Device::CPU>;
/// the FFT oversampler of real arrays (correlation surfaces) on the CPU
using FFTOverSamplerR2R = OverSampler<OversamplingMethod::FFT, cpu::real_type, Device::CPU>;

/**
 * Create a batched 2D complex fft plan
 * @param nx,ny the size of the images
 * @param nImages the number of images
 * @param in,out the input and output arrays
 * @param sign the sign of the exponent (FFTW_FORWARD or FFTW_BACKWARD)
 */
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
 * Constructor of the FFT oversampler of complex arrays
 * @param inNX,inNY the size of the input images
 * @param outNX,outNY the size of the output (oversampled) images
 * @param nImages the number of images
 * @param stream_ the stream (not used)
 */
FFTOverSamplerC2C::OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    // set up work arrays
    workIn = array_type<complex_type>(inNX, inNY, nImages);
    workOut = array_type<complex_type>(outNX, outNY, nImages);

    // set up fft plans (out-of-place), with temporary arrays in place of the input/output images
    array_type<complex_type> planIn(inNX, inNY, nImages);
    array_type<complex_type> planOut(outNX, outNY, nImages);
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    // exp(+i) to the frequency domain and exp(-i) back, as the isce3 (v1) pycuampcor:
    // the Nyquist frequency of even lengths is then positive (see cuArraysFFTPaddingMany)
    forwardPlan = planMany2D(inNX, inNY, nImages, planIn.data_handle(), workIn.data_handle(), FFTW_BACKWARD);
    backwardPlan = planMany2D(outNX, outNY, nImages, workOut.data_handle(), planOut.data_handle(), FFTW_FORWARD);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void FFTOverSamplerC2C::execute(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut)
{
    // FFT to frequency domain (the input of the out-of-place transform is not written,
    // fftw takes it as non-const)
    PYCUAMPCOR_FFTW(execute_dft)(forwardPlan, fftwCast(const_cast<complex_type *>(imagesIn.data_handle())),
        fftwCast(workIn.data_handle()));
    // padding zeros in the middle
    cuArraysFFTPaddingMany(workIn, workOut, stream);
    // iFFT back to time domain
    PYCUAMPCOR_FFTW(execute_dft)(backwardPlan, fftwCast(workOut.data_handle()), fftwCast(imagesOut.data_handle()));
}

/**
 * Release the fftw plans and the work arrays
 */
FFTOverSamplerC2C::~OverSampler() = default;


/**
 * Constructor of the FFT oversampler of real arrays
 * @param inNX,inNY the size of the input images
 * @param outNX,outNY the size of the output (oversampled) images
 * @param nImages the number of images
 * @param stream_ the stream (not used)
 */
FFTOverSamplerR2R::OverSampler(int inNX, int inNY, int outNX, int outNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    workSizeIn = array_type<complex_type>(inNX, inNY, nImages);
    // work arrays for one image
    workColumns = array_type<complex_type>(outNX, inNY, 1);
    workPadded = array_type<complex_type>(outNX, outNY, 1);
    workSizeOut = array_type<complex_type>(outNX, outNY, 1);

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
        forwardPlan = planMany2D(inNX, inNY, nImages, workSizeIn.data_handle(), workSizeIn.data_handle(), FFTW_BACKWARD);
        // inverse fft along the columns, from workColumns to the non-zero columns of workPadded
        for(int k = 0; k < 2; k++)
            backwardPlanColumns[k] = (nColumns[k] == 0) ? nullptr : PYCUAMPCOR_FFTW(plan_many_dft)(1, &outNX, nColumns[k],
                fftwCast(workColumns.data_handle() + columnIn[k]), nullptr, inNY, 1,
                fftwCast(workPadded.data_handle() + columnOut[k]), nullptr, outNY, 1,
                FFTW_FORWARD, fftwFlags | FFTW_PRESERVE_INPUT);
        // inverse fft along the rows, out-of-place to keep the zero columns of workPadded
        backwardPlanRows = PYCUAMPCOR_FFTW(plan_many_dft)(1, &outNY, outNX,
            fftwCast(workPadded.data_handle()), nullptr, 1, outNY,
            fftwCast(workSizeOut.data_handle()), nullptr, 1, outNY,
            FFTW_FORWARD, fftwFlags | FFTW_PRESERVE_INPUT);
    }

    // the zeros padded in the middle, set once (only the non-zero parts are written in execute)
    workColumns.setZero(stream);
    workPadded.setZero(stream);
}

/**
 * Execute fft oversampling
 * @param[in] imagesIn input batch of images
 * @param[out] imagesOut output batch of images
 */
void FFTOverSamplerR2R::execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut)
{
    // forward fft of all images
    cuArraysCopyPadded(imagesIn, workSizeIn, stream);
    PYCUAMPCOR_FFTW(execute)(forwardPlan);

    const int inNX = workSizeIn.height(), inNY = workSizeIn.width();
    const int outNX = workSizeOut.height();
    const int imagesOutNX = imagesOut.height(), imagesOutNY = imagesOut.width();
    const ArraysView<complex_type> columns = workColumns;
    const ArraysView<const complex_type> oversampled = workSizeOut;
    // fftw doesn't normalize
    const real_type factor = real_type(1)/workSizeIn.arraySize();
    for(int imageIdx = 0; imageIdx < workSizeIn.count(); imageIdx++) {
        // pad zeros in the middle of the columns (as cuArraysFFTPaddingMany), with the spectrum
        // rows moved to the top and bottom; the zero rows are not changed
        const ArraysView<const complex_type> spectrum = std::as_const(workSizeIn).view().array(imageIdx);
        for(int x1 = 0; x1 < inNX; x1++) {
            const int x2 = (x1 < (inNX+1)/2) ? x1 : outNX - inNX + x1;
            for(int y = 0; y < inNY; y++)
                columns(x2, y) = spectrum(x1, y)*factor;
        }
        // inverse fft along the columns, then along the rows
        for(int k = 0; k < 2; k++)
            if(backwardPlanColumns[k])
                PYCUAMPCOR_FFTW(execute)(backwardPlanColumns[k]);
        PYCUAMPCOR_FFTW(execute)(backwardPlanRows);
        // take the real part
        const ArraysView<real_type> imageOut = imagesOut.array(imageIdx);
        for(int x = 0; x < imagesOutNX; x++)
            for(int y = 0; y < imagesOutNY; y++)
                imageOut(x, y) = oversampled(x, y).x;
    }
}

/**
 * Release the fftw plans and the work arrays
 */
FFTOverSamplerR2R::~OverSampler() = default;

} // namespace
// end of file
