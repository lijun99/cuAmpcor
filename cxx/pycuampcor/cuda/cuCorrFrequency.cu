/*
 * @file  cuCorrFrequency.cu
 * @brief A class performs cross correlation in frequency domain
 */

#include "cuCorrFrequency.h"
#include "float2.h"
#include "cuArraysCopy.h"
#include "device.h"
#include "cudaError.h"
#include "logging.h"
#include "cuArrays.h"

namespace pycuampcor::cuda {

// coef*conj(image1)*image2, element-wise, in image1 (defined below)
static void cuArraysElementMultiplyConjugate(ArraysView<complex_type> image1,
    ArraysView<const complex_type> image2, real_type coef, cudaStream_t stream);

/*
 * cuFreqCorrelator Constructor
 * @param imageNX height of each image
 * @param imageNY width of each image
 * @param nImages number of images in the batch
 * @param stream CUDA stream
 */
cuFreqCorrelator::cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_)
{

    int imageSize = imageNX*imageNY;
    int fImageSize = imageNX*(imageNY/2+1);
    int n[NRANK] ={imageNX, imageNY};

    // set up fft plans
#ifdef CUAMPCOR_DOUBLE
    forwardPlan.planMany(NRANK, n,
        nullptr, 1, imageSize,
        nullptr, 1, fImageSize,
        CUFFT_D2Z, nImages);
    backwardPlan.planMany(NRANK, n,
        nullptr, 1, fImageSize,
        nullptr, 1, imageSize,
        CUFFT_Z2D, nImages);
#else
    forwardPlan.planMany(NRANK, n,
        nullptr, 1, imageSize,
        nullptr, 1, fImageSize,
        CUFFT_R2C, nImages);
    backwardPlan.planMany(NRANK, n,
        nullptr, 1, fImageSize,
        nullptr, 1, imageSize,
        CUFFT_C2R, nImages);
#endif
    stream = stream_;
    cufftSetStream(forwardPlan, stream);
    cufftSetStream(backwardPlan, stream);

    // set up work arrays
    workFM = cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workFS = cuArrays<complex_type>(imageNX, (imageNY/2+1), nImages);
    workT = cuArrays<real_type>(imageNX, imageNY, nImages);
}

/// destructor
cuFreqCorrelator::~cuFreqCorrelator() = default;

/**
 * Execute the cross correlation
 * @param[in] templates the reference windows
 * @param[in] images the search windows
 * @param[out] results the correlation surfaces
 */

void cuFreqCorrelator::execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results)
{
    // pad the reference windows to the the size of search windows
    cuArraysCopyPadded(templates, workT, stream);
    // the input of the (out-of-place) forward FFT is not written (cufft takes it as non-const)
    real_type *imagesData = const_cast<real_type *>(images.data_handle());
    // forward fft to frequency domain
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecD2Z(forwardPlan, workT.data_handle(), workFM.data_handle()));
    checkCudaErrors(cufftExecD2Z(forwardPlan, imagesData, workFS.data_handle()));
#else
    checkCudaErrors(cufftExecR2C(forwardPlan, workT.data_handle(), workFM.data_handle()));
    checkCudaErrors(cufftExecR2C(forwardPlan, imagesData, workFS.data_handle()));
#endif

    // cufft doesn't normalize, so manually get the image size for normalization
    real_type coef = 1.0/(images.arraySize());
    // multiply reference with secondary windows in frequency domain
    cuArraysElementMultiplyConjugate(workFM, workFS, coef, stream);
    // backward fft to get correlation surface in time domain
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2D(backwardPlan, workFM.data_handle(), workT.data_handle()));
#else
    checkCudaErrors(cufftExecC2R(backwardPlan, workFM.data_handle(), workT.data_handle()));
#endif
    // extract to get proper size of correlation surface
    cuArraysCopyExtract(workT, results, make_int2(0, 0), stream);
    // all done
}

// a = a^* * b
inline __device__ complex_type cuMulConj(complex_type a, complex_type b, real_type f)
{
    return make_complex_type(f*(a.x*b.x + a.y*b.y), f*(-a.y*b.x + a.x*b.y));
}

// cuda kernel for cuArraysElementMultiplyConjugate
__global__ void cudaKernel_elementMulConjugate(ArraysView<complex_type> ainout,
    ArraysView<const complex_type> bin, int size, real_type coef)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        ainout [idx] = cuMulConj(ainout[idx], bin[idx], coef);
    }
}

/**
 * Perform multiplication of coef*Conjugate[image1]*image2 for each element
 * @param[in,out] image1, the first image
 * @param[in] image2, the secondary image
 * @param[in] coef, usually the normalization factor
 */
static void cuArraysElementMultiplyConjugate(ArraysView<complex_type> image1,
    ArraysView<const complex_type> image2, real_type coef, cudaStream_t stream)
{
    int size = image1.size();
    int threadsperblock = NTHREADS;
    int blockspergrid = divUp(size, threadsperblock);
    cudaKernel_elementMulConjugate<<<blockspergrid, threadsperblock, 0, stream>>>(image1, image2, size, coef );
    checkCudaAsyncErrors();
}
//end of file

} // namespace