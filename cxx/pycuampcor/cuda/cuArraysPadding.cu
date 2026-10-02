/*
 * @file  cuArraysPadding.cu
 * @brief Utilities for padding zeros to cuArrays for FFT (zero padding in the middle)
 */

#include "cuAmpcorUtil.h"
#include "float2.h"

namespace pycuampcor::cuda {


// cuda kernel for zero padding for FFT oversampling,
// for both even and odd sequences
// @param[in] image1 input images
// @param[in,out] image2 output images - memset to 0 in prior
// @note siding Nyquist frequency for even length with negative frequency (as numpy/FFTW frequencies);
//       splitting it between the positive and negative frequencies makes no difference for real signals
//       (the real part is taken after the inverse transform), and the Nyquist component of oversampled,
//       deramped SLCs is negligible
// for even N - positive f[0, ..., N/2-1],
//              zeros 0...0,
//              negative f[N/2], f[N/2+1, ..., N-1]
// for odd N - positive f[0, ..., (N-1)/2],
//             zeros 0...0,
//             negative f [(N+1)/2, ..., N-1]
__global__ void cuArraysPaddingMany_kernel(
    const complex_type *image1, const int height1, const int width1, const int size1,
    complex_type *image2, const int height2, const int width2, const int size2, const real_type factor )
{
    // thread indices are for input image1
    int x1 = threadIdx.x + blockDim.x*blockIdx.x;
    int y1 = threadIdx.y + blockDim.y*blockIdx.y;
    int imageIdx =  blockIdx.z;

    //  ensure threads are in the range of image1
    if (x1 >= height1 || y1 >= width1)
        return;

    // determine the quadrants
    // divup the length to be consistent with both even and odd lengths
    int x2 = (x1 < (height1+1)/2) ? x1 : height2 - height1 + x1;
    int y2 = (y1 < (width1+1)/2) ? y1 : width2 - width1 + y1;
    image2[IDX2R(x2, y2, width2)+imageIdx*size2]
            = image1[IDX2R(x1, y1, width1)+imageIdx*size1]*factor;
    return;
}

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 */
void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, cudaStream_t stream)
{
    int ThreadsPerBlock = NTHREADS2D;
    // up to IDIVUP(dim, 2) for odd-length sequences
    int BlockPerGridx = IDIVUP (image1->height, ThreadsPerBlock);
    int BlockPerGridy = IDIVUP (image1->width, ThreadsPerBlock);
    dim3 dimBlock(ThreadsPerBlock, ThreadsPerBlock, 1);
    dim3 dimGrid(BlockPerGridx, BlockPerGridy, image1->count);

    checkCudaErrors(cudaMemsetAsync(image2->devData, 0, image2->getByteSize(),stream));
    real_type factor = real_type(1)/image1->size;
    cuArraysPaddingMany_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        image1->devData, image1->height, image1->width, image1->size,
        image2->devData, image2->height, image2->width, image2->size, factor);
    getLastCudaError("cuArraysPadding_kernel");
}
//end of file










} // namespace