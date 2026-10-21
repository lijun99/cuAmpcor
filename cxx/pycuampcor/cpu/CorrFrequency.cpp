/**
 * @file  CorrFrequency.cpp
 * @brief The cross-correlator of the windows in the frequency domain on the CPU
 */

#include "CorrFrequency.h"
#include "float2.h"
#include "ArraysCopy.h"
#include "Arrays.h"

namespace pycuampcor::cpu {

/**
 * Multiply conj(image1) by image2 and coef, element-wise, in image1 (defined below)
 */
static void cuArraysElementMultiplyConjugate(ArraysView<complex_type> image1,
    ArraysView<const complex_type> image2, real_type coef, stream_t stream);

/**
 * The mutex to guard the fftw planner (shared by all fftw users in this backend)
 */
std::mutex & fftwPlannerMutex()
{
    static std::mutex mutex;
    return mutex;
}

} // namespace

namespace pycuampcor {

// the definitions use the CPU package: its types, arrays and fftw plans
using namespace cpu;

/// the cross-correlator in the frequency domain on the CPU
using FrequencyCorrelator = Correlator<Algorithm::Frequency, cpu::real_type, Device::CPU>;

/**
 * Constructor of the cross-correlator in the frequency domain
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 * @param stream_ (not used)
 */
FrequencyCorrelator::Correlator(int imageNX, int imageNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    int imageSize = imageNX*imageNY;
    int fImageSize = imageNX*(imageNY/2+1);
    int n[2] ={imageNX, imageNY};

    // set up work arrays
    workFM = array_type<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFS = array_type<complex_type>(imageNX, (imageNY/2+1), nImages);
    workT = array_type<real_type>(imageNX, imageNY, nImages);

    // set up fft plans
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    forwardPlan = PYCUAMPCOR_FFTW(plan_many_dft_r2c)(2, n, nImages,
        workT.data_handle(), nullptr, 1, imageSize,
        fftwCast(workFM.data_handle()), nullptr, 1, fImageSize,
        fftwFlags);
    backwardPlan = PYCUAMPCOR_FFTW(plan_many_dft_c2r)(2, n, nImages,
        fftwCast(workFM.data_handle()), nullptr, 1, fImageSize,
        workT.data_handle(), nullptr, 1, imageSize,
        fftwFlags);
}

/**
 * Release the fftw plans and the work arrays
 */
FrequencyCorrelator::~Correlator() = default;

/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */
void FrequencyCorrelator::execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results)
{
    // pad the reference windows to the the size of search windows
    cuArraysCopyPadded(templates, workT, stream);
    // forward fft to frequency domain (the input of the out-of-place r2c transform is not written,
    // fftw takes it as non-const)
    PYCUAMPCOR_FFTW(execute_dft_r2c)(forwardPlan, workT.data_handle(), fftwCast(workFM.data_handle()));
    PYCUAMPCOR_FFTW(execute_dft_r2c)(forwardPlan, const_cast<real_type *>(images.data_handle()),
        fftwCast(workFS.data_handle()));
    // fftw doesn't normalize, so manually get the image size for normalization
    real_type coef = 1.0/(images.arraySize());
    // multiply reference with secondary windows in frequency domain
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    // backward fft to get correlation surface in time domain
    PYCUAMPCOR_FFTW(execute_dft_c2r)(backwardPlan, fftwCast(workFM.data_handle()), workT.data_handle());
    // extract to get proper size of correlation surface
    cuArraysCopyExtract(workT, results, make_int2(0, 0), stream);
    // all done
}

} // namespace

namespace pycuampcor::cpu {

/**
 * Perform multiplication of coef*Conjugate[image1]*image2 for each element
 * @param[in,out] image1, the first image
 * @param[in] image2,the secondary image
 * @param[in] coef,usually the normalization factor
 */
static void cuArraysElementMultiplyConjugate(ArraysView<complex_type> image1,
    ArraysView<const complex_type> image2, real_type coef, stream_t)
{
    const size_t size = image1.size();
    complex_type *data1 = image1.data_handle();
    const complex_type *data2 = image2.data_handle();
    for(size_t idx = 0; idx < size; idx++) {
        const complex_type a = data1[idx];
        const complex_type b = data2[idx];
        data1[idx] = make_complex_type(coef*(a.x*b.x + a.y*b.y), coef*(-a.y*b.x + a.x*b.y));
    }
}

} // namespace
// end of file
