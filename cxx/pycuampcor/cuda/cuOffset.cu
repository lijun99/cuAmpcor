/**
 * @file  cuOffset.cu
 * @brief Utilities used to determine the offset field
 *
 */

// my module dependencies
#include "cuOffset.h"
#include "float2.h"
#include "device.h"
#include "cudaError.h"

namespace pycuampcor::cuda {

/**
 * Find the max between two elements of the shared memory, kept (with its location) at gid
 */
inline static __device__ void maxPairReduce(real_type* maxval, int* maxloc,
      size_t gid, size_t strideid)
{
    if(maxval[gid] < maxval[strideid]) {
        maxval[gid] = maxval[strideid];
        maxloc[gid] = maxloc[strideid];
    }
}

/**
 * Max reduction for a 2d image (device function), by a block of BLOCKSIZE threads
 * start from (start.x, start.y) in a rectangle range (range.x, range.y)
 * start + range <= imageSize
 * @param shval,shloc the shared memory for the values and the (flattened) locations; the result is
 *   in shval[0], shloc[0]
 */
template<const int BLOCKSIZE>
__device__ void max_reduction_2d(const real_type* const image,
    const int2 imageSize,
    const int2 start, const int2 range,
    real_type* shval,
    int* shloc)
{
    int tid = threadIdx.x;
    shval[tid] = -REAL_MAX;
    // default to the first pixel in case no value is larger (e.g., all NaNs)
    shloc[tid] = idx2r(start.x, start.y, imageSize.y);

    // reduction for intra-block elements
    // i.e., for elements with i, i+BLOCKSIZE, i+2*BLOCKSIZE ...
    for(int gid = tid; gid < range.x*range.y; gid+=blockDim.x)
    {
        // gid is the flattened pixel id in the search range
        // get the pixel 2d coordinate in whole image
        int idx = start.x + gid / range.y;
        int idy = start.y + gid % range.y;
        // get the flattened 1d coordinate
        int pixelId = idx2r(idx, idy, imageSize.y);
        real_type pixelValue = image[pixelId];
        if(shval[tid] < pixelValue) {
            shval[tid] = pixelValue;
            shloc[tid] = pixelId;
        }
    }
    __syncthreads();

    // reduction within a block with shared memory
    if (BLOCKSIZE >=1024){ if (tid < 512) { maxPairReduce(shval, shloc, tid, tid + 512); } __syncthreads(); }
    if (BLOCKSIZE >=512) { if (tid < 256) { maxPairReduce(shval, shloc, tid, tid + 256); } __syncthreads(); }
    if (BLOCKSIZE >=256) { if (tid < 128) { maxPairReduce(shval, shloc, tid, tid + 128); } __syncthreads(); }
    if (BLOCKSIZE >=128) { if (tid < 64 ) { maxPairReduce(shval, shloc, tid, tid + 64 ); } __syncthreads(); }
    // reduction within a warp; lanes of a warp are not executed in lockstep (since Volta),
    // __syncwarp separates the reads and writes of each step
    if (tid < 32)
    {
        real_type val = shval[tid];
        int loc = shloc[tid];
        for (int offset = 32; offset > 0; offset /= 2) {
            const real_type other = shval[tid + offset];
            const int otherLoc = shloc[tid + offset];
            if (val < other) {
                val = other;
                loc = otherLoc;
            }
            __syncwarp();
            shval[tid] = val;
            shloc[tid] = loc;
            __syncwarp();
        }
    }
    __syncthreads();
}

/**
 * Kernel for cuArraysMaxloc2D: one block per 2D array (image), the max value and its location
 */
template <const int BLOCKSIZE>
__global__ void  cudaKernel_maxloc2D(ArraysView<const real_type> images, ArraysView<int2> maxloc,
    ArraysView<real_type> maxval, const int2 start, const int2 range)
{
    __shared__ real_type shval[BLOCKSIZE];
    __shared__ int shloc[BLOCKSIZE];

    int imageIdx = blockIdx.x;
    if(imageIdx >= images.count()) return;

    // get the starting pointer for this image
    const int2 imageSize = make_int2(images.height(), images.width());
    const real_type *image = images.array(imageIdx).data_handle();

    max_reduction_2d<BLOCKSIZE>(image, imageSize, start, range, shval, shloc);

    // thread 0 contains the final result, convert it to 2d coordinate
    if (threadIdx.x == 0) {
        maxloc[imageIdx] = make_int2(shloc[0]/imageSize.y, shloc[0]%imageSize.y);
        maxval[imageIdx] = shval[0];
    }
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] start starting search pixel
 * @param[in] range search range
 * @param[in] stream cudaStream
 * @note This routine is overloaded with the routine without maxval
 */
void cuArraysMaxloc2D(ArraysView<const real_type> images,
                      const int2 start, const int2 range,
                      ArraysView<int2> maxloc, ArraysView<real_type> maxval, cudaStream_t stream)
{
    cudaKernel_maxloc2D<NTHREADS><<<images.count(), NTHREADS, 0, stream>>>
        (images, maxloc, maxval, start, range);
    checkCudaAsyncErrors();
}

/**
 * Find both the maximum value and the location for a batch of 2D images
 * @param[in] images input batch of images
 * @param[out] maxval arrays to hold the max values
 * @param[out] maxloc arrays to hold the max locations
 * @param[in] stream cudaStream
 */
void cuArraysMaxloc2D(ArraysView<const real_type> images,
                      ArraysView<int2> maxloc, ArraysView<real_type> maxval, cudaStream_t stream)
{
    // if no start and range are provided, use the whole image
    int2 start = make_int2(0, 0);
    int2 imageSize = make_int2(images.height(), images.width());
    cudaKernel_maxloc2D<NTHREADS><<<images.count(), NTHREADS, 0, stream>>>
        (images, maxloc, maxval, start, imageSize);
    checkCudaAsyncErrors();
}

/**
 * CUDA kernel for cuSubPixelOffset: one thread per window
 */
__global__ void cuSubPixelOffset_kernel(ArraysView<const int2> offsetInit,
                                        ArraysView<const int2> offsetZoomIn,
                                        ArraysView<real2_type> offsetFinal, const int size,
                                        const int2 initOrigin, const real_type initRatio,
                                        const int2 zoomInOrigin, const real_type zoomInRatio)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) return;
    offsetFinal[idx].x = initRatio*(offsetInit[idx].x-initOrigin.x) + zoomInRatio*(offsetZoomIn[idx].x - zoomInOrigin.x);
    offsetFinal[idx].y = initRatio*(offsetInit[idx].y-initOrigin.y) + zoomInRatio*(offsetZoomIn[idx].y - zoomInOrigin.y);
}

/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 */
void cuSubPixelOffset(ArraysView<const int2> offsetInit,
    ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal,
    const int2 initOrigin, const int initFactor,
    const int2 zoomInOrigin, const int zoomInFactor,
    cudaStream_t stream)
{
    int size = offsetInit.size();

    // GPU performs multiplication faster
    real_type initRatio = 1.0f/static_cast<real_type>(initFactor);
    real_type zoomInRatio = 1.0f/static_cast<real_type>(zoomInFactor);

    cuSubPixelOffset_kernel<<<divUp(size, NTHREADS), NTHREADS, 0, stream>>>
        (offsetInit, offsetZoomIn, offsetFinal, size, initOrigin, initRatio, zoomInOrigin, zoomInRatio);
    checkCudaAsyncErrors();

}

/**
 * CUDA kernel for cuSubPixelOffset2Pass: one thread per window
 */
__global__ void cuSubPixelOffset2Pass_kernel(ArraysView<const int2> offsetInit,
                                        ArraysView<const int2> offsetZoomIn,
                                        ArraysView<real2_type> offsetFinal,
                                        const real_type OSratio,
                                        const real_type xoffset, const real_type yoffset, const int size)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if (idx >= size) return;
    offsetFinal[idx].x = OSratio*(offsetZoomIn[idx].x ) + offsetInit[idx].x  - xoffset;
    offsetFinal[idx].y = OSratio*(offsetZoomIn[idx].y ) + offsetInit[idx].y - yoffset;
}

/**
 * Determine the final offset value
 * @param[in] offsetInit max location (adjusted to the starting location for extraction) determined from
 *   the cross-correlation before oversampling, in dimensions of pixel
 * @param[in] offsetZoomIn max location from the oversampled cross-correlation surface
 * @param[out] offsetFinal the combined offset value
 * @param[in] OverSampleRatioZoomin the correlation surface oversampling factor
 * @param[in] OverSampleRatioRaw the oversampling factor of reference/secondary windows before cross-correlation
 * @param[in] xHalfRangeInit the original half search range along x, to be subtracted
 * @param[in] yHalfRangeInit the original half search range along y, to be subtracted
 *
 * 1. Cross-correlation is performed at first for the un-oversampled data with a larger search range.
 *   The secondary window is then extracted to a smaller size (a smaller search range) around the max location.
 *   The extraction starting location (offsetInit) - original half search range (xHalfRangeInit, yHalfRangeInit)
 *        = pixel size offset
 * 2. Reference/secondary windows are then oversampled by OverSampleRatioRaw, and cross-correlated.
 * 3. The correlation surface is further oversampled by OverSampleRatioZoomin.
 *    The overall oversampling factor is OverSampleRatioZoomin*OverSampleRatioRaw.
 *    The max location in oversampled correlation surface (offsetZoomIn) / overall oversampling factor
 *        = subpixel offset
 *    Final offset =  pixel size offset +  subpixel offset
 */
void cuSubPixelOffset2Pass(ArraysView<const int2> offsetInit, ArraysView<const int2> offsetZoomIn,
    ArraysView<real2_type> offsetFinal,
    int OverSampleRatioZoomin, int OverSampleRatioRaw,
    int xHalfRangeInit,  int yHalfRangeInit,
    cudaStream_t stream)
{
    int size = offsetInit.size();
    real_type OSratio = real_type(1)/(OverSampleRatioZoomin*OverSampleRatioRaw);
    real_type xoffset = xHalfRangeInit ;
    real_type yoffset = yHalfRangeInit ;

    cuSubPixelOffset2Pass_kernel<<<divUp(size, NTHREADS), NTHREADS, 0, stream>>>
        (offsetInit, offsetZoomIn, offsetFinal, OSratio, xoffset, yoffset, size);
    checkCudaAsyncErrors();

}

/**
 * CUDA device function to compute the start of extraction and the shift of center
 * @return the start (x) and the shift (y)
 */
static inline __device__ int2 dev_adjustOffset(
    const int oldRange, const int newRange, const int maxloc)
{
    // determine the starting point around the maxloc
    // oldRange is the half search window size, e.g., = 32
    // newRange is the half extract size, e.g., = 4
    // maxloc is in range [0, 64]
    // we want to extract \pm 4 centered at maxloc
    // Examples:
    // 1. maxloc = 40: we set start=maxloc-newRange=36, and extract [36,44), shift=0
    // 2. maxloc = 2, start=-2: we set start=0, shift=-2,
    //   (shift means the max is -2 from the extracted center 4)
    // 3. maxloc =64, start=60: set start=56, shift = 4
    //   (shift means the max is 4 from the extracted center 60).

    // shift the max location by -newRange to find the start
    int start = maxloc - newRange;
    // if start is within the range, the max location will be in the center
    int shift = 0;
    // right boundary
    int rbound = 2*(oldRange-newRange);
    if(start<0)     // if exceeding the limit on the left
    {
        // set start at 0 and record the shift of center
        // (negative, the max location is on the left of the extracted center)
        shift = start;
        start = 0;
    }
    else if(start > rbound ) // if exceeding the limit on the right
    {
        //
        shift = start-rbound;
        start = rbound;
    }
    return make_int2(start, shift);
}

/**
 * CUDA kernel for cuDetermineSecondaryExtractOffset: one thread per window
 */
__global__ void cudaKernel_determineSecondaryExtractOffset(ArraysView<int2> maxLoc, ArraysView<int2> shift,
    const size_t nImages, int xOldRange, int yOldRange, int xNewRange, int yNewRange)
{
    int imageIndex = threadIdx.x + blockDim.x *blockIdx.x; //image index
    if (imageIndex < nImages)
    {
        // get the starting pixel (stored back to maxloc) and shift
        int2 result = dev_adjustOffset(xOldRange, xNewRange, maxLoc[imageIndex].x);
        maxLoc[imageIndex].x = result.x;
        shift[imageIndex].x = result.y;
        result = dev_adjustOffset(yOldRange, yNewRange, maxLoc[imageIndex].y);
        maxLoc[imageIndex].y = result.x;
        shift[imageIndex].y = result.y;
    }
}

/**
 * Determine the secondary window extract offset from the max location
 * @param[in] xOldRange,yOldRange are (half) search ranges in first step
 * @param[in] xNewRange,yNewRange are (half) search range
 *
 * After the first run of cross-correlation, with a larger search range,
 *  We now choose a smaller search range around the max location for oversampling.
 *  This procedure is used to determine the starting pixel locations for extraction.
 */
void cuDetermineSecondaryExtractOffset(ArraysView<int2> maxLoc, ArraysView<int2> maxLocShift,
    int xOldRange, int yOldRange, int xNewRange, int yNewRange, cudaStream_t stream)
{
    int threadsperblock=NTHREADS;
    int size = maxLoc.size();
    int blockspergrid=divUp(size, threadsperblock);
    cudaKernel_determineSecondaryExtractOffset<<<blockspergrid, threadsperblock, 0, stream>>>
        (maxLoc, maxLocShift, size, xOldRange, yOldRange, xNewRange, yNewRange);
    checkCudaAsyncErrors();
}

// end of file

} // namespace