/**
 * @file  cuCorrNormalization.cu
 * @brief Utilities to normalize the correlation surface
 *
 * The mean and variance of the normalization factor can be computed from the
 *   cumulative/prefix sum (or sum area table) s(u,v), and s2(u,v).
 * We follow the algorithm by Evghenii Gaburov, Tim Idzenga, Willem Vermin, in the nxcor package.
 * 1. Iterate over rows and for each row, the cumulative sum for elements in the row
 *    is computed as c_row(u,v) = \sum_(v'<v) f(u, v')
 *    and we keep track of the sum of area of width Ny, i.e.,
 *         c(u,v) = \sum_{u'<=u} [c_row(u', v+Ny) - c_row(u', v)],
 *         or c(u,v) = c(u-1, v) + [c_row(u, v+Ny) - c_row(u, v)]
 * 2. When row reaches the window height u=Nx-1,
 *    c(u,v) provides the sum of area for the first batch of windows sa(0,v).
 * 3. proceeding to row = u+1, we compute both c_row(u+1, v) and c_row(u-Nx, v)
 *    i.e., we add the sum from new row and remove the sum from the first row in c(u,v):
 *    c(u+1,v)= c(u,v) + [c_row(u+1,v+Ny)-c_row(u+1, v)] - [c_row(u-Nx, v+Ny)-c_row(u-Nx, v)].
 * 4. Iterate 3. over the rest rows, and c(u,v) provides the sum of areas for new row of windows.
 *
 */

#include "cuCorrNormalization.h"
#include "float2.h"
#include "device.h"
#include "cudaError.h"

namespace pycuampcor::cuda {

/**
 * Sum reduction within a block (of Nthreads >= 64 threads), with the result returned to all threads
 *
 * The following implementation is compatible with sm_20 and above; newer architectures may
 * support faster implementations, such as warp shuffle, cooperative groups.
 * @param sum the value of this thread
 * @param shmem the shared memory to use, of Nthreads values
 */
template <const int Nthreads>
__device__ real_type sumReduceBlock(real_type sum, real_type *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();

    if (Nthreads >=1024) { if (tid < 512) { shmem[tid] += shmem[tid + 512]; } __syncthreads(); }
    if (Nthreads >= 512) { if (tid < 256) { shmem[tid] += shmem[tid + 256]; } __syncthreads(); }
    if (Nthreads >= 256) { if (tid < 128) { shmem[tid] += shmem[tid + 128]; } __syncthreads(); }
    if (Nthreads >= 128) { if (tid <  64) { shmem[tid] += shmem[tid +  64]; } __syncthreads(); }
    // reduction within a warp; lanes of a warp are not executed in lockstep (since Volta),
    // __syncwarp separates the reads and writes of each step
    if (tid < 32)
    {
        real_type v = shmem[tid];
        for (int offset = 32; offset > 0; offset /= 2) {
            v += shmem[tid + offset];
            __syncwarp();
            shmem[tid] = v;
            __syncwarp();
        }
    }

    __syncthreads();
    const real_type result = shmem[0];
    // all threads have to read the result before shmem is reused (e.g., by another reduction)
    __syncthreads();
    return result;
}

/**
 * Cuda kernel to compute and subtract the mean value from the images (one block per image)
 */
template<const int Nthreads>
__global__ void cuArraysSubtractMean_kernel(ArraysView<real_type> images, real_type invSize)
{
    __shared__ real_type shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= images.count()) return;

    const int       imageIdx = bid;
    const int imageSize = images.arraySize();
    real_type   *imageD = images.array(imageIdx).data_handle();

    // compute the sum
    real_type sum  = 0.0;
    for (int i = tid; i < imageSize; i += Nthreads)
            sum += imageD[i];
    sum = sumReduceBlock<Nthreads>(sum, shmem);

    // compute the mean
    const real_type mean = sum * invSize;
    // subtract the mean from each pixel
    for (int i = tid; i < imageSize; i += Nthreads)
            imageD[i] -= mean;
}

/**
 * Compute and subtract mean values from images
 * @param[in,out] images Input/Output images
 * @param[in] stream cudaStream
 */
void cuArraysSubtractMean(ArraysView<real_type> images, cudaStream_t stream)
{
    const dim3 grid(images.count(), 1, 1);
    const int imageSize = images.arraySize();
    const real_type invSize = real_type(1)/imageSize;

    cuArraysSubtractMean_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images, invSize);
    checkCudaAsyncErrors();
}

/**
 * Cuda kernel to compute the sums of the squares and the counts of valid pixels of the extracted
 * correlation surfaces (Minyan), one block per image
 */
template<const int Nthreads>
__global__ void cuArraysSumCorr_kernel(ArraysView<const real_type> images,
    ArraysView<const int> imagesValid, ArraysView<real_type> imagesSum,
    ArraysView<int> imagesValidCount)
{
    __shared__ real_type shmem[Nthreads];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (bid >= images.count()) return;

    const int imageIdx = bid;
    const int imageSize = images.arraySize();
    const real_type*    imageD = images.array(imageIdx).data_handle();
    const int*      imageValidD = imagesValid.array(imageIdx).data_handle();

    real_type sum  = 0.0f;
    real_type count = 0;

    for (int i = tid; i < imageSize; i += Nthreads) {
            sum += imageD[i] * imageD[i];
            count += imageValidD[i];
    }

    sum = sumReduceBlock<Nthreads>(sum, shmem);
    count = sumReduceBlock<Nthreads>(count, shmem);

    if(tid ==0) {
        imagesSum[bid] = sum;
        imagesValidCount[bid] = static_cast<int>(count);
    }
}

/**
 * Compute the variance of images (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum variance
 * @param[out] imagesValidCount count of total valid pixels
 * @param[in] stream cudaStream
 */
void cuArraysSumCorr(ArraysView<const real_type> images, ArraysView<const int> imagesValid,
    ArraysView<real_type> imagesSum, ArraysView<int> imagesValidCount, cudaStream_t stream)
{
    const dim3 grid(images.count(), 1, 1);

    cuArraysSumCorr_kernel<NTHREADS> <<<grid,NTHREADS,0,stream>>>(images, imagesValid,
        imagesSum, imagesValidCount);
    checkCudaAsyncErrors();
}

/**
 * Intra-block inclusive prefix sum, in shmem, with 2^Nthreads2 threads
 * @param sum the value of this thread
 */
template<int Nthreads2>
__device__ void inclusive_prefix_sum(real_type sum, real_type *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = sum;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Nthreads2; i++)
    {
        const int offset = 1 << i;
        if (tid >= offset) sum += shmem[tid - offset];
        __syncthreads();
        shmem[tid] = sum;
        __syncthreads();
    }
}

/**
 * Prefix sum of pixel value and pixel value^2
 * @return the sums of v and v^2 over the stride threads from this one
 */
template<const int Nthreads2>
__device__ real2_type partialSums(const real_type v, real_type* shmem, const int stride)
{
    const int tid = threadIdx.x;

    real_type *shMem  = shmem + 1;
    real_type *shMem2 = shMem + 1 + (1 << Nthreads2);

    inclusive_prefix_sum<Nthreads2>(v,   shMem);
    inclusive_prefix_sum<Nthreads2>(v*v, shMem2);
    const real_type Sum  = shMem [tid-1 + stride] - shMem [tid-1];
    const real_type Sum2 = shMem2[tid-1 + stride] - shMem2[tid-1];
    // all threads have to read the sums before shmem is reused (by the next call)
    __syncthreads();
    return make_real2(Sum, Sum2);
}

/**
 * Cuda kernel for cuCorrNormalizeFixed, with the sums over the secondary windows updated row by
 * row (see the notes above); one block of 2^Nthreads2 threads per image
 * @param templateCoeff the inverse of the size of the reference windows
 */
template<const int Nthreads2>
__global__ void cuCorrNormalize_kernel(ArraysView<const real_type> templates,
    ArraysView<const real_type> images, ArraysView<real_type> results, real_type templateCoeff)
{
    const int Nthreads = 1<<Nthreads2;
    __shared__ real_type shmem[Nthreads*3];

    const int tid = threadIdx.x;
    const int imageIdx = blockIdx.z;
    if (imageIdx >= images.count()) return;

    const int templateNX = templates.height(), templateNY = templates.width();
    const int templateSize = templates.arraySize();
    const int imageNY = images.width(), imageSize = images.arraySize();
    const int resultNY = results.width();

    const real_type *   imageD =    images.array(imageIdx).data_handle();
    const real_type *templateD = templates.array(imageIdx).data_handle();
    real_type *  resultD =   results.array(imageIdx).data_handle();

    // template sum^2
    real_type templateSum2 = 0.0;
    for (int i = tid; i < templateSize; i += Nthreads)
        {
            const real_type t = templateD[i];
            templateSum2 += t*t;
        }
    templateSum2 = sumReduceBlock<Nthreads>(templateSum2, shmem);
    __syncthreads();

    // reset shared memory value
    shmem[tid] = shmem[tid + Nthreads] = shmem[tid + 2*Nthreads] = 0.0;
    __syncthreads();

    // perform the prefix sum and sum^2 for secondary window
    // see notes above
    real_type imageSum  = 0.0;
    real_type imageSum2 = 0.0;
    int iaddr = 0;
    const int windowSize = templateNX*imageNY;
    // iterative till reaching the templateNX row of the secondary window
    // or the first row of correlation surface may be computed
    while (iaddr < windowSize)
    {
        // cum sum for each row with a width=templateNY
        const real2_type res = partialSums<Nthreads2>(imageD[iaddr + tid], shmem, templateNY);
        // add to the total, which keeps track of the sum of area for each window
        imageSum  += res.x;
        imageSum2 += res.y;
        // move to next row
        iaddr     += imageNY;
    }
    // row reaches the end of first batch of windows
    // normalize the first row of the correlation surface
    if (tid < resultNY)
    {
        // normalizing factor
        const real_type norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
        // normalize the correlation surface
        resultD[tid] *= rsqrt(norm2 + EPSILON);
    }
    // iterative over the rest rows
    while (iaddr < imageSize)
    {
        // the prefix sum of the row removed is recomputed, to be subtracted
        const real2_type res1 = partialSums<Nthreads2>(imageD[iaddr-windowSize + tid], shmem, templateNY);
        // the prefix sum of the new row, to be added
        const real2_type res2 = partialSums<Nthreads2>(imageD[iaddr            + tid], shmem, templateNY);
        imageSum  += res2.x - res1.x;
        imageSum2 += res2.y - res1.y;
        // move to next row
        iaddr     += imageNY;
        // normalize the correlation surface
        if (tid < resultNY)
        {
            const int ix = iaddr/imageNY; // get row index
            const int addr = (ix-templateNX)*resultNY; // get the correlation surface row index
            const real_type norm2 = (imageSum2 - imageSum*imageSum*templateCoeff)*templateSum2;
            resultD[addr + tid] *= rsqrt(norm2 + EPSILON);
        }
    }
}

/// the base 2 logarithm of N, for the block sizes of cuCorrNormalizeFixed
template<int N> struct Log2;
template<> struct Log2<64> { static const int value = 6; };
template<> struct Log2<128> { static const int value = 7; };
template<> struct Log2<256> { static const int value = 8; };
template<> struct Log2<512> { static const int value = 9; };
template<> struct Log2<1024> { static const int value = 10; };

/**
 * Normalize the correlation surfaces, with blocks of Size threads (summing in shared memory)
 */
template<int Size>
void cuCorrNormalizeFixed(ArraysView<real_type> correlation, ArraysView<const real_type> reference,
    ArraysView<const real_type> secondary, cudaStream_t stream)
{
    const int nImages = correlation.count();
    const dim3 grid(1, 1, nImages);
    const real_type invReferenceSize = real_type(1)/reference.arraySize();
    cuCorrNormalize_kernel<Log2<Size>::value><<<grid, Size, 0, stream>>>(reference, secondary,
        correlation, invReferenceSize);
    checkCudaAsyncErrors();
}

template void cuCorrNormalizeFixed<64>(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream);
template void cuCorrNormalizeFixed<128>(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream);
template void cuCorrNormalizeFixed<256>(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream);
template void cuCorrNormalizeFixed<512>(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream);
template void cuCorrNormalizeFixed<1024>(ArraysView<real_type> correlation,
    ArraysView<const real_type> reference, ArraysView<const real_type> secondary, cudaStream_t stream);

// end of file

} // namespace