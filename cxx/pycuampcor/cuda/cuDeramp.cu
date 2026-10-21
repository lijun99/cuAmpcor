/**
 * @file  cuDeramp.cu
 * @brief Derampling a batch of 2D complex images with GPU
 *
 * A phase ramp is equivalent to a frequency shift in frequency domain,
 *   which needs to be removed (deramping) in order to move the band center
 *   to zero. This is necessary before oversampling a complex signal.
 * Method Complex: each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 *   The average is weighted by the amplitudes (coherence).
 * Other methods: skip deramping
 *
 */

#include "cuDeramp.h"
#include "Options.h"
#include "float2.h"
#include "cudaError.h"
#include "device.h"
#include "debug.h"

namespace pycuampcor::cuda {

/**
 * Store the real2_type value x at the index i of the shared memory s
 * @note cuda does not have a good support on vector structure, e.g. complex type (float2 or double2);
 *   shared memory holds real_type values; real2_type values are stored as two halves, a block apart;
 *   copyToShared, copyFromShared and addInShared operate real2_type objects through real_type
 */
inline static __device__ void copyToShared(real_type *s, const int i, const real2_type x, const int block)
{ s[i] = x.x; s[i+block] = x.y; }

/**
 * Load the real2_type value x from the index i of the shared memory s (the halves a block apart)
 */
inline static __device__ void copyFromShared(real2_type &x, real_type *s, const int i, const int block)
{ x.x = s[i]; x.y = s[i+block]; }

/**
 * Add the real2_type value at the index i+j of the shared memory s to the value at the index i
 */
inline static __device__ void addInShared(real_type *s, const int i, const int j, const int block)
{ s[i] += s[i+j]; s[i+block] += s[i+j+block];}

/**
 * Sum reduction of real2_type values within a block of nthreads threads
 * @param[in,out] sum the value of the thread as input, the sum over the block as output (all threads)
 * @param shmem the shared memory of 2*nthreads real_type values
 */
template <const int nthreads>
__device__ void complexSumReduceBlock(real2_type& sum, real_type *shmem)
{
    const int tid = threadIdx.x;
    copyToShared(shmem, tid, sum, nthreads);
    __syncthreads();

    if (nthreads >=1024) { if (tid < 512) { addInShared(shmem, tid, 512, nthreads); } __syncthreads(); }
    if (nthreads >= 512) { if (tid < 256) { addInShared(shmem, tid, 256, nthreads); } __syncthreads(); }
    if (nthreads >= 256) { if (tid < 128) { addInShared(shmem, tid, 128, nthreads); } __syncthreads(); }
    if (nthreads >= 128) { if (tid <  64) { addInShared(shmem, tid,  64, nthreads); } __syncthreads(); }
    // reduction within a warp; lanes of a warp are not executed in lockstep (since Volta),
    // __syncwarp separates the reads and writes of each step
    if (tid < 32)
    {
        real_type vx = shmem[tid], vy = shmem[tid + nthreads];
        for (int offset = 32; offset > 0; offset /= 2) {
            vx += shmem[tid + offset];
            vy += shmem[tid + offset + nthreads];
            __syncwarp();
            shmem[tid] = vx;
            shmem[tid + nthreads] = vy;
            __syncwarp();
        }
    }
    __syncthreads();
    copyFromShared(sum, shmem, 0, nthreads);
    // all threads have to read the result before shmem is reused (e.g., by another reduction)
    __syncthreads();
}

/**
 * CUDA kernel for cuLinearDeramp: one block per window (blockIdx.x), its threads striding over the
 * pixels to sum the phase differences and to remove the phase ramp
 * @param[in] normCoef the normalization coefficient (unused)
 * @param[in] axis 0: deramp along x-axis; 1: deramp along y-axis; others: deramp along both axes
 */
template<const int nthreads>
__global__ void cuLinearDeramp_kernel(ArraysView<real2_type> images, const real_type normCoef, int axis)
{
    __shared__ real_type shmem[2*nthreads];
    int pixelIdx, pixelIdxX, pixelIdxY;
    const int imageNX = images.height(), imageNY = images.width();
    const int imageSize = images.arraySize();

    const int bid = blockIdx.x;
    if(bid >= images.count()) return;
    const ArraysView<real2_type> image = images.array(bid);
    const int tid = threadIdx.x;

    // compute phase difference along y direction
    double phaseY = 0.0;
    if (axis != 0) {
        real2_type phaseDiffY  = make_real2(0.0, 0.0);
        for (int i = tid; i < imageSize; i += nthreads) {
            pixelIdxY = i % imageNY;
            if(pixelIdxY < imageNY -1) {
                pixelIdx = i;
                real2_type cprod = complexMulConj( image[pixelIdx], image[pixelIdx+1]);
                phaseDiffY += cprod;
            }
        }
        complexSumReduceBlock<nthreads>(phaseDiffY, shmem);
        // phaseDiffY *= normCoef;
        phaseY=atan2(phaseDiffY.y, phaseDiffY.x);
    }

    // compute phase difference along x direction
    double phaseX = 0.0;
    if (axis != 1) {
        real2_type phaseDiffX  = make_real2(0.0, 0.0);
        for (int i = tid; i < imageSize; i += nthreads)  {
            pixelIdxX = i / imageNY;
            if(pixelIdxX < imageNX -1) {
                pixelIdx = i;
                real2_type cprod = complexMulConj(image[i], image[i+imageNY]);
                phaseDiffX += cprod;
            }
        }
        complexSumReduceBlock<nthreads>(phaseDiffX, shmem);

        phaseX = atan2(phaseDiffX.y, phaseDiffX.x);
    }

#ifdef CUAMPCOR_DEBUG
    // output the phase ramp in debug mode
    if (threadIdx.x==0)
    {
        printf("debug linear deramp az: %g rg: %g\n", phaseX, phaseY);
    }
#endif

    for (int i = tid; i < imageSize; i += nthreads)
    {
        pixelIdxX = i/imageNY;
        pixelIdxY = i%imageNY;
        double phase = pixelIdxX*phaseX + pixelIdxY*phaseY;
        double phase_sin, phase_cos;
        sincos(phase, &phase_sin, &phase_cos);
        image[i] = make_real2(
            image[i].x*phase_cos - image[i].y*phase_sin,
            image[i].x*phase_sin + image[i].y*phase_cos);
    }
}

/**
 * Deramp complex signals (the method DerampMethod::Complex): each signal is decomposed into real
 * and imaginary parts, and the average phase shift is obtained as atan(\sum imag / \sum real)
 * @param[in,out] images input/output complex signals
 * @param[in] stream cuda stream
 */
static void cuLinearDeramp(ArraysView<real2_type> images, const int axis, cudaStream_t stream)
{

    const dim3 grid(images.count());
    const int imageSize = images.arraySize();
    const real_type invSize = real_type(1)/imageSize;

    if(imageSize <=64)
        cuLinearDeramp_kernel<64> <<<grid, 64, 0, stream>>>(images, invSize, axis);
    else if(imageSize <=128)
        cuLinearDeramp_kernel<128> <<<grid, 128, 0, stream>>>(images, invSize, axis);
    else if(imageSize <=256)
        cuLinearDeramp_kernel<256> <<<grid, 256, 0, stream>>>(images, invSize, axis);
    else
        cuLinearDeramp_kernel<512> <<<grid, 512, 0, stream>>>(images, invSize, axis);
    checkCudaAsyncErrors();

}

/**
 * Remove the phase ramp (phaseX, phaseY per pixel down and across) from a window, the threads of
 * the block striding over its pixels
 */
template<const int nthreads>
__device__ void removeRamp(ArraysView<real2_type> image, double phaseX, double phaseY)
{
    const int width = image.width();
    for (int i = threadIdx.x; i < image.arraySize(); i += nthreads) {
        double phase = (i/width)*phaseX + (i%width)*phaseY;
        double phase_sin, phase_cos;
        sincos(phase, &phase_sin, &phase_cos);
        image[i] = make_real2(
            image[i].x*phase_cos - image[i].y*phase_sin,
            image[i].x*phase_sin + image[i].y*phase_cos);
    }
}

/**
 * CUDA kernel for cuDerampByReference: one block per window (blockIdx.x), the phase ramp of the
 * reference window estimated as in cuLinearDeramp_kernel, and removed from both windows
 * @param[in] axis 0: deramp along x-axis; 1: deramp along y-axis; others: deramp along both axes
 */
template<const int nthreads>
__global__ void cuDerampByReference_kernel(ArraysView<real2_type> reference,
    ArraysView<real2_type> secondary, int axis)
{
    __shared__ real_type shmem[2*nthreads];
    const int imageNX = reference.height(), imageNY = reference.width();
    const int imageSize = reference.arraySize();

    const int bid = blockIdx.x;
    if(bid >= reference.count()) return;
    const ArraysView<real2_type> image = reference.array(bid);
    const int tid = threadIdx.x;

    // the phase difference along y
    double phaseY = 0.0;
    if (axis != 0) {
        real2_type phaseDiffY  = make_real2(0.0, 0.0);
        for (int i = tid; i < imageSize; i += nthreads) {
            if(i % imageNY < imageNY - 1)
                phaseDiffY += complexMulConj(image[i], image[i+1]);
        }
        complexSumReduceBlock<nthreads>(phaseDiffY, shmem);
        phaseY = atan2(phaseDiffY.y, phaseDiffY.x);
    }

    // the phase difference along x
    double phaseX = 0.0;
    if (axis != 1) {
        real2_type phaseDiffX  = make_real2(0.0, 0.0);
        for (int i = tid; i < imageSize; i += nthreads) {
            if(i / imageNY < imageNX - 1)
                phaseDiffX += complexMulConj(image[i], image[i+imageNY]);
        }
        complexSumReduceBlock<nthreads>(phaseDiffX, shmem);
        phaseX = atan2(phaseDiffX.y, phaseDiffX.x);
    }

    // remove the ramp from the reference window and from the secondary window
    removeRamp<nthreads>(image, phaseX, phaseY);
    removeRamp<nthreads>(secondary.array(bid), phaseX, phaseY);
}

/**
 * Remove the phase ramps of the reference windows from both the reference windows and the
 * secondary windows (in place), along the axis, with the method: the ramp of each reference window
 * is removed from it and from its secondary window, which keeps their relative phase
 * @note only DerampMethod::Complex deramps; the other methods do nothing
 */
void cuDerampByReference(DerampMethod method, ArraysView<complex_type> reference,
    ArraysView<complex_type> secondary, DerampAxis axis, cudaStream_t stream)
{
    if (method != DerampMethod::Complex)
        return;
    const dim3 grid(reference.count());
    const int imageSize = reference.arraySize();
    const int ax = static_cast<int>(axis);
    if(imageSize <=64)
        cuDerampByReference_kernel<64> <<<grid, 64, 0, stream>>>(reference, secondary, ax);
    else if(imageSize <=128)
        cuDerampByReference_kernel<128> <<<grid, 128, 0, stream>>>(reference, secondary, ax);
    else if(imageSize <=256)
        cuDerampByReference_kernel<256> <<<grid, 256, 0, stream>>>(reference, secondary, ax);
    else
        cuDerampByReference_kernel<512> <<<grid, 512, 0, stream>>>(reference, secondary, ax);
    checkCudaAsyncErrors();
}

/**
 * Remove the phase ramps of the complex windows (in place), along the axis, with the method
 * @note only DerampMethod::Complex deramps (with cuLinearDeramp); the other methods do nothing
 */
void cuDeramp(DerampMethod method, ArraysView<complex_type> images, DerampAxis axis, cudaStream_t stream)
{
    // the axis values (down 0, across 1, both 2) are those of cuLinearDeramp
    if (method == DerampMethod::Complex)
        cuLinearDeramp(images, static_cast<int>(axis), stream);
}

// end of file

} // namespace