/*
 * @file  cuCorrFrequency.cpp
 * @brief A class performs cross correlation in frequency domain (CPU backend)
 */

#include "cuCorrFrequency.h"
#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

// the mutex to guard the fftw planner (shared by all fftw users in this backend)
std::mutex & fftwPlannerMutex()
{
    static std::mutex mutex;
    return mutex;
}

/*
 * cuFreqCorrelator Constructor
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 * @param stream (not used)
 */
cuFreqCorrelator::cuFreqCorrelator(int imageNX, int imageNY, int nImages, stream_t stream_)
    : stream(stream_)
{
    int imageSize = imageNX*imageNY;
    int fImageSize = imageNX*(imageNY/2+1);
    int n[2] ={imageNX, imageNY};

    // set up work arrays
    workFM = new cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFM->allocate();
    workFS = new cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFS->allocate();
    workT = new cuArrays<real_type> (imageNX, imageNY, nImages);
    workT->allocate();

    // set up fft plans
    std::lock_guard<std::mutex> lock(fftwPlannerMutex());
    forwardPlan = PYCUAMPCOR_FFTW(plan_many_dft_r2c)(2, n, nImages,
        workT->devData, NULL, 1, imageSize,
        fftwCast(workFM->devData), NULL, 1, fImageSize,
        PYCUAMPCOR_FFTW_FLAGS);
    backwardPlan = PYCUAMPCOR_FFTW(plan_many_dft_c2r)(2, n, nImages,
        fftwCast(workFM->devData), NULL, 1, fImageSize,
        workT->devData, NULL, 1, imageSize,
        PYCUAMPCOR_FFTW_FLAGS);
}

/// destructor
cuFreqCorrelator::~cuFreqCorrelator()
{
    {
        std::lock_guard<std::mutex> lock(fftwPlannerMutex());
        PYCUAMPCOR_FFTW(destroy_plan)(forwardPlan);
        PYCUAMPCOR_FFTW(destroy_plan)(backwardPlan);
    }
    delete workFM;
    delete workFS;
    delete workT;
}

/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */
void cuFreqCorrelator::execute(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results)
{
    // pad the reference windows to the the size of search windows
    cuArraysCopyPadded(templates, workT, stream);
    // forward fft to frequency domain
    PYCUAMPCOR_FFTW(execute_dft_r2c)(forwardPlan, workT->devData, fftwCast(workFM->devData));
    PYCUAMPCOR_FFTW(execute_dft_r2c)(forwardPlan, images->devData, fftwCast(workFS->devData));
    // fftw doesn't normalize, so manually get the image size for normalization
    real_type coef = 1.0/(images->size);
    // multiply reference with secondary windows in frequency domain
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    // backward fft to get correlation surface in time domain
    PYCUAMPCOR_FFTW(execute_dft_c2r)(backwardPlan, fftwCast(workFM->devData), workT->devData);
    // extract to get proper size of correlation surface
    cuArraysCopyExtract(workT, results, make_int2(0, 0), stream);
    // all done
}

/**
 * Perform multiplication of coef*Conjugate[image1]*image2 for each element
 * @param[in,out] image1, the first image
 * @param[in] image2, the secondary image
 * @param[in] coef, usually the normalization factor
 */
void cuArraysElementMultiplyConjugate(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, real_type coef, stream_t)
{
    const size_t size = image1->getSize();
    for(size_t idx = 0; idx < size; idx++) {
        const complex_type a = image1->devData[idx];
        const complex_type b = image2->devData[idx];
        image1->devData[idx] = make_complex_type(coef*(a.x*b.x + a.y*b.y), coef*(-a.y*b.x + a.x*b.y));
    }
}

} // namespace
// end of file
