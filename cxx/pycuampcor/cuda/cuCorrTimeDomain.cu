/**
 * @file  cuCorrTimeDomain.cu
 * @brief Correlation between two sets of images in time domain
 *
 * This code is adapted from the nxcor package.
 */

#include "cuCorrTimeDomain.h"
#include "float2.h"
#include "cudaError.h"

namespace pycuampcor::cuda {

/**
 * CUDA kernel for cuCorrTimeDomain: one block per pair of windows (blockIdx.x) and NPT rows of its
 * correlation surface, starting at the row blockIdx.y*NPT
 * @note the block of nthreads threads (at least the width of the secondary window) loads the rows of
 *   the template and of the secondary window to shared memory; the thread tx computes the column tx
 *   of the NPT rows, with the template columns split among q groups of threads, summed at the end
 */
template<const int nthreads, const int NPT>
__global__ void cuArraysCorrTime_kernel(ArraysView<const real_type> templates,
    ArraysView<const real_type> images, ArraysView<real_type> results)
{
    __shared__ real_type shmem[nthreads*(1+NPT)];
    const int tid = threadIdx.x;
    const int bid =  blockIdx.x;
    const int  yc =  blockIdx.y*NPT;

    const int templateNY = templates.width(), templateSize = templates.arraySize();
    const int imageNY = images.width();
    const int resultNY = results.width(), resultSize = results.arraySize();

    const int       imageIdx = bid;
    const real_type *   imageD =    images.array(imageIdx).data_handle() + tid;
    const real_type *templateD = templates.array(imageIdx).data_handle() + tid;
     real_type *  resultD =   results.array(imageIdx).data_handle();

    const int q  = min(nthreads/resultNY, 4);
    const int nt = nthreads/q;
    const int ty = threadIdx.x / nt;
    const int tx = threadIdx.x - nt * ty;

    const int templateNYq = templateNY/q;
    const int jbeg = templateNYq * ty;
    const int jend = ty+1 >= q ? templateNY : templateNYq + jbeg;

    real_type *shTemplate = shmem;
    real_type *shImage    = shmem + nthreads;
    real_type *shImage1   = shImage + tx;

    real_type corrCoeff[NPT];
    for (int k = 0; k < NPT; k++)
        corrCoeff[k] = 0.0;

    int iaddr = yc*imageNY;

    real_type img[NPT];
    for (int k = 0; k < NPT-1; k++, iaddr += imageNY)
        img[k] = imageD[iaddr];
    for (int taddr = 0; taddr < templateSize; taddr += templateNY, iaddr += imageNY)
    {
        shTemplate[tid] = templateD[taddr];
        img     [NPT-1] =    imageD[iaddr];
        for (int k = 0; k < NPT; k++)
            shImage[tid + nthreads*k] = img[k];
        for (int k = 0; k < NPT-1; k++)
            img[k] = img[k+1];
        __syncthreads();

        if (tx < resultNY && ty < q)
        {
#pragma unroll 8
            for (int j = jbeg; j < jend; j++)
                for (int k = 0; k < NPT; k++)
                    corrCoeff[k] += shTemplate[j]*shImage1[j + nthreads*k];
        }
        __syncthreads();
    }

    for (int k = 0; k < NPT; k++)
        shmem[tid + nthreads*k] = corrCoeff[k];
    __syncthreads();

    for (int j = tx + nt; j < nthreads; j += nt)
        for (int k = 0; k < NPT; k++)
            corrCoeff[k] += shmem[j + nthreads*k];
    __syncthreads();

    if (tid < resultNY)
    {
        int raddr = yc*resultNY + tid;
        for (int k = 0; k < NPT; k++, raddr += resultNY)
            if (raddr < resultSize)
                resultD[raddr] = corrCoeff[k];
    }
}

/**
 * Perform cross correlation in time domain
 * @param[in] templates Reference images
 * @param[in] images Secondary images
 * @param[out] results Output correlation surface
 * @param[in] stream cudaStream
 */
void cuCorrTimeDomain(ArraysView<const real_type> templates,
               ArraysView<const real_type> images,
               ArraysView<real_type> results,
               cudaStream_t stream)
{
    /* compute correlation matrix */
    const int nImages = images.count();
    const int imageNY = images.width();

#ifdef CUAMPCOR_DOUBLE
    // static shared memory is limited to 48K; the same NPT must be used in both host and device passes
    // (so __CUDA_ARCH__ can't be used to choose NPT here)
    const int NPT = 4;
#else
    const int NPT = 8;
#endif

    // one block per window and per NPT rows of its correlation surface
    const dim3 grid(nImages, (results.height()-1)/NPT+1, 1);
    if      (imageNY <=   64) {
        cuArraysCorrTime_kernel<  64,NPT><<<grid,  64, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  128)  {
        cuArraysCorrTime_kernel< 128,NPT><<<grid, 128, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  192) {
        cuArraysCorrTime_kernel< 192,NPT><<<grid, 192, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  256) {
        cuArraysCorrTime_kernel< 256,NPT><<<grid, 256, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  384) {
        cuArraysCorrTime_kernel< 384,NPT><<<grid, 384, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  512) {
        cuArraysCorrTime_kernel< 512,NPT><<<grid, 512, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  640) {
        cuArraysCorrTime_kernel< 640,NPT><<<grid, 640, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  768) {
        cuArraysCorrTime_kernel< 768,NPT><<<grid, 768, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <=  896) {
        cuArraysCorrTime_kernel< 896,NPT><<<grid, 896, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else if (imageNY <= 1024) {
        cuArraysCorrTime_kernel<1024,NPT><<<grid,1024, 0, stream>>>(templates, images, results);
        checkCudaAsyncErrors();
    }
    else {
        throw std::invalid_argument("The (oversampled) window size along the across direction "
            + std::to_string(imageNY) + " should be smaller than 1024.");
    }
}

} // namespace

namespace pycuampcor {

/// the cross-correlator in the time domain on the CUDA device
using TimeCorrelator = Correlator<Algorithm::Time, cuda::real_type, Device::CUDA>;

/**
 * Constructor of the cross-correlator in the time domain
 * @param stream_ the CUDA stream of the kernels
 */
TimeCorrelator::Correlator(int, int, int, cudaStream_t stream_)
    : stream(stream_)
{
}

/**
 * Compute the correlation surfaces of the reference windows with the secondary windows
 */
void TimeCorrelator::execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results)
{
    cuda::cuCorrTimeDomain(templates, images, results, stream);
}

} // namespace
// end of file
