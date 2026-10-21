/**
 * @file  cuCorrCoherence.cu
 * @brief The coherent cross-correlator of the complex windows on the CUDA device
 */

#include "cuCorrCoherence.h"
#include "float2.h"
#include "device.h"
#include "cudaError.h"
#include "cuArrays.h"
#include "indexing.h"

namespace pycuampcor::cuda {

/// the number of threads of the kernels with one block per window
constexpr int COHERENCE_THREADS = 256;

/**
 * Sum of the values of the threads of a block (of COHERENCE_THREADS threads), returned to all
 * threads
 * @param shmem the shared memory of COHERENCE_THREADS values
 */
__device__ real_type blockSum(real_type value, real_type *shmem)
{
    const int tid = threadIdx.x;
    shmem[tid] = value;
    __syncthreads();
    for (int stride = COHERENCE_THREADS/2; stride > 0; stride /= 2) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    const real_type sum = shmem[0];
    // all threads read the sum before the shared memory is reused
    __syncthreads();
    return sum;
}

/**
 * The reference windows with their means subtracted, zero padded to the size of the secondary
 * windows, and the sums of their squared magnitudes (one block per window)
 */
__global__ void coherenceTemplate_kernel(ArraysView<const complex_type> templates,
    ArraysView<complex_type> padded, ArraysView<real_type> referenceSum2)
{
    __shared__ real_type shmem[COHERENCE_THREADS];
    const int k = blockIdx.x;
    const int h = templates.height(), w = templates.width(), n = h*w;
    const int H = padded.height(), W = padded.width();
    const ArraysView<const complex_type> t = templates.array(k);

    // the mean
    real_type sx = 0, sy = 0;
    for (int i = threadIdx.x; i < n; i += COHERENCE_THREADS) {
        sx += t[i].x;
        sy += t[i].y;
    }
    const real_type mx = blockSum(sx, shmem)/n;
    const real_type my = blockSum(sy, shmem)/n;

    // the padded windows, and the sum of the squared magnitudes
    real_type s2 = 0;
    for (int i = threadIdx.x; i < H*W; i += COHERENCE_THREADS) {
        const int row = i / W, col = i % W;
        complex_type v = make_complex_type(0, 0);
        if (row < h && col < w) {
            v = make_complex_type(t(row, col).x - mx, t(row, col).y - my);
            s2 += v.x*v.x + v.y*v.y;
        }
        padded(k, row, col) = v;
    }
    s2 = blockSum(s2, shmem);
    if (threadIdx.x == 0)
        referenceSum2[k] = s2;
}

/**
 * The products conj(a) b scaled by coef, in a (one thread per element)
 */
__global__ void coherenceMulConj_kernel(complex_type *a, const complex_type *b, size_t size,
    real_type coef)
{
    const size_t idx = threadIdx.x + size_t(blockIdx.x)*blockDim.x;
    if (idx < size) {
        const complex_type x = a[idx], y = b[idx];
        a[idx] = make_complex_type(coef*(x.x*y.x + x.y*y.y), coef*(x.x*y.y - x.y*y.x));
    }
}

/**
 * The (inclusive) sum area tables of the secondary windows: their real and imaginary parts, and
 * their squared magnitudes (one block per window: the rows, then the columns)
 */
__global__ void coherenceSat_kernel(ArraysView<const complex_type> images, ArraysView<real_type> sat,
    ArraysView<real_type> satImag, ArraysView<real_type> sat2)
{
    const int k = blockIdx.x;
    const int H = images.height(), W = images.width();
    for (int row = threadIdx.x; row < H; row += blockDim.x) {
        real_type sx = 0, sy = 0, s2 = 0;
        for (int col = 0; col < W; col++) {
            const complex_type v = images(k, row, col);
            sx += v.x; sy += v.y; s2 += v.x*v.x + v.y*v.y;
            sat(k, row, col) = sx;
            satImag(k, row, col) = sy;
            sat2(k, row, col) = s2;
        }
    }
    __syncthreads();
    for (int col = threadIdx.x; col < W; col += blockDim.x)
        for (int row = 1; row < H; row++) {
            sat(k, row, col) += sat(k, row - 1, col);
            satImag(k, row, col) += satImag(k, row - 1, col);
            sat2(k, row, col) += sat2(k, row - 1, col);
        }
}

/**
 * The sum over a template of h x w at the shift (tx, ty), from a sum area table
 */
__device__ real_type areaSum(const ArraysView<const real_type> &s, int k, int tx, int ty, int h, int w)
{
    const real_type bottomright = s(k, tx + h - 1, ty + w - 1);
    const real_type topleft = (tx > 0 && ty > 0) ? s(k, tx - 1, ty - 1) : real_type(0);
    const real_type topright = (tx > 0) ? s(k, tx - 1, ty + w - 1) : real_type(0);
    const real_type bottomleft = (ty > 0) ? s(k, tx + h - 1, ty - 1) : real_type(0);
    return bottomright + topleft - topright - bottomleft;
}

/**
 * The squared coherence at each shift: the squared magnitude of the correlation over the norms
 * (one thread per pixel of the surfaces, blockIdx.z the window)
 */
__global__ void coherenceNormalize_kernel(ArraysView<real_type> results,
    ArraysView<const complex_type> correlation, ArraysView<const real_type> referenceSum2,
    ArraysView<const real_type> sat, ArraysView<const real_type> satImag,
    ArraysView<const real_type> sat2, int h, int w)
{
    const int k = blockIdx.z;
    const int tx = threadIdx.x + blockDim.x*blockIdx.x;
    const int ty = threadIdx.y + blockDim.y*blockIdx.y;
    if (tx < results.height() && ty < results.width()) {
        const real_type sx = areaSum(sat, k, tx, ty, h, w);
        const real_type sy = areaSum(satImag, k, tx, ty, h, w);
        const real_type norm2 = (areaSum(sat2, k, tx, ty, h, w) - (sx*sx + sy*sy)/(h*w))*referenceSum2[k];
        const complex_type c = correlation(k, tx, ty);
        results(k, tx, ty) = (c.x*c.x + c.y*c.y)/(norm2 + EPSILON);
    }
}

} // namespace

namespace pycuampcor {

// the definitions use the CUDA package: its types, arrays and cufft plans
using namespace cuda;

/// the coherent cross-correlator on the CUDA device
using CoherenceCorrelator = Correlator<Algorithm::Coherence, cuda::complex_type, Device::CUDA>;

/**
 * Constructor of the coherent cross-correlator: the cufft plan and the work arrays
 * @param imageNX,imageNY the size of the secondary windows
 * @param nImages the number of windows
 * @param stream_ the cuda stream
 */
CoherenceCorrelator::Correlator(int imageNX, int imageNY, int nImages, cudaStream_t stream_)
    : stream(stream_)
{
    const int imageSize = imageNX*imageNY;
    int n[NRANK] = {imageNX, imageNY};
#ifdef CUAMPCOR_DOUBLE
    plan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, imageSize, CUFFT_Z2Z, nImages);
#else
    plan.planMany(NRANK, n, nullptr, 1, imageSize, nullptr, 1, imageSize, CUFFT_C2C, nImages);
#endif
    cufftSetStream(plan, stream);
    workT = cuArrays<complex_type>(imageNX, imageNY, nImages);
    workS = cuArrays<complex_type>(imageNX, imageNY, nImages);
    referenceSum2 = cuArrays<real_type>(1, 1, nImages);
    sat = cuArrays<real_type>(imageNX, imageNY, nImages);
    satImag = cuArrays<real_type>(imageNX, imageNY, nImages);
    sat2 = cuArrays<real_type>(imageNX, imageNY, nImages);
}

/**
 * Release the cufft plan and the work arrays
 */
CoherenceCorrelator::~Correlator() = default;

/**
 * Compute the squared coherence surfaces
 * @param[in] templates the reference windows
 * @param[in] images the secondary windows
 * @param[out] results the squared coherence surfaces (secondary - reference + 1 each way)
 */
void CoherenceCorrelator::execute(ArraysView<const complex_type> templates,
    ArraysView<const complex_type> images, ArraysView<real_type> results)
{
    const int count = images.count();
    // the padded reference windows with their means subtracted, and their norms
    coherenceTemplate_kernel<<<count, COHERENCE_THREADS, 0, stream>>>(templates, workT, referenceSum2);
    checkCudaAsyncErrors();
    // the spectra (the input of the forward transform of the secondary windows is not written)
    checkCudaErrors(cudaMemcpyAsync(workS.data_handle(), images.data_handle(),
        images.size()*sizeof(complex_type), cudaMemcpyDeviceToDevice, stream));
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(plan, workT.data_handle(), workT.data_handle(), CUFFT_FORWARD));
    checkCudaErrors(cufftExecZ2Z(plan, workS.data_handle(), workS.data_handle(), CUFFT_FORWARD));
#else
    checkCudaErrors(cufftExecC2C(plan, workT.data_handle(), workT.data_handle(), CUFFT_FORWARD));
    checkCudaErrors(cufftExecC2C(plan, workS.data_handle(), workS.data_handle(), CUFFT_FORWARD));
#endif
    // conj(T) S, back to the correlations sum conj(t) s at each shift (cufft doesn't normalize)
    const size_t size = workT.size();
    coherenceMulConj_kernel<<<divUp(int(size), NTHREADS), NTHREADS, 0, stream>>>(
        workT.data_handle(), workS.data_handle(), size, real_type(1)/images.arraySize());
    checkCudaAsyncErrors();
#ifdef CUAMPCOR_DOUBLE
    checkCudaErrors(cufftExecZ2Z(plan, workT.data_handle(), workT.data_handle(), CUFFT_INVERSE));
#else
    checkCudaErrors(cufftExecC2C(plan, workT.data_handle(), workT.data_handle(), CUFFT_INVERSE));
#endif
    // the sum area tables of the secondary windows, and the squared coherence
    coherenceSat_kernel<<<count, COHERENCE_THREADS, 0, stream>>>(images, sat, satImag, sat2);
    checkCudaAsyncErrors();
    const dim3 block(NTHREADS2D, NTHREADS2D, 1);
    const dim3 grid(divUp(results.height(), NTHREADS2D), divUp(results.width(), NTHREADS2D), count);
    coherenceNormalize_kernel<<<grid, block, 0, stream>>>(results, workT, referenceSum2,
        sat, satImag, sat2, templates.height(), templates.width());
    checkCudaAsyncErrors();
}

} // namespace
// end of file
