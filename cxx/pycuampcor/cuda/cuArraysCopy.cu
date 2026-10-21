/**
 * @file  cuArraysCopy.cu
 * @brief Utilities for copying/converting images to different format
 *
 * All methods are declared in cuArraysCopy.h
 * cuArraysCopyToBatch*WithOffset to extract a batch of windows from the raw image
 *   various implementations include:
 *   1. varying offsets, as start pixels for windows
 *   2. complex to complex, usually
 *   3. complex to (amplitude,0), for TOPS
 *   4. real to complex, for real images
 * cuArraysCopyExtract to extract(shrink in size) from a batch of windows to another batch
 *   overloaded for different data types
 * cuArraysCopyInsert to insert a batch of windows (smaller in size) to another batch
 *   overloaded for different data types
 * cuArraysCopyPadded to insert a batch of windows to another batch while padding 0s for rest elements
 *   used for fft oversampling
 *   see also cuArraysPadding.cu for other zero-padding utilities
 * cuArraysR2C cuArraysC2R cuArraysAbs to convert between different data types
 */

// dependencies
#include "cuArraysCopy.h"
#include "device.h"
#include "cudaError.h"
#include "float2.h"

namespace pycuampcor::cuda {

/// the pixel of a chunk as a window pixel: the complex value, or a real value (as complex)
struct ToComplex {
    /// the complex pixel in the internal complex type
    __device__ complex_type operator()(const image_complex_type &v) const { return make_complex_type(v.x, v.y); }
    /// the real pixel as a complex value with a zero imaginary part
    __device__ complex_type operator()(const image_real_type &v) const { return complex_type{v}; }
};
/// the pixel of a chunk as a window pixel: its amplitude (as complex)
struct ToAmplitude {
    /// the amplitude of the complex pixel, with a zero imaginary part
    __device__ complex_type operator()(const image_complex_type &v) const { return make_complex_type(complexAbs(v), 0.0); }
};

/**
 * Kernel for cuArraysCopyToBatch*WithOffset: one thread per window pixel (outx, outy) of the window
 * blockIdx.z, filled with 0 outside the chunk
 * @param op the conversion of a chunk pixel to a window pixel (ToComplex or ToAmplitude)
 */
template<typename T_in, typename Op>
__global__ void cuArraysCopyToBatchWithOffset_kernel(ArraysView<const T_in> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetX, ArraysView<const int> offsetY,
    Op op)
{
    // get image index
    int idxImage = blockIdx.z;
    // check the image index within range
    if(idxImage >= windows.count()) return;
    // get the output pixel location
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;
     // check the output location within range (due to cuda threads)
    if(outx >= windows.height() || outy >= windows.width()) return;
    // find the input pixel location
    int inx = offsetX[idxImage] + outx;
    int iny = offsetY[idxImage] + outy;
    // check whether the location is within the input image range
    if(inx>=0 && inx<chunk.height() && iny>=0 && iny<chunk.width())
        windows(idxImage, outx, outy) = op(chunk(inx, iny)); //true, copy
    else
        windows(idxImage, outx, outy) = make_complex_type(0.0, 0.0); //false, fill with 0
}

/**
 * Launch cuArraysCopyToBatchWithOffset_kernel, one block of threads per 16 x 16 window pixels
 */
template<typename T_in, typename Op>
static void copyToBatchWithOffset(ArraysView<const T_in> chunk, ArraysView<complex_type> windows,
    ArraysView<const int> offsetH, ArraysView<const int> offsetW, Op op, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 blockSize(nthreads, nthreads, 1);
    dim3 gridSize(divUp(windows.height(),nthreads), divUp(windows.width(),nthreads), windows.count());
    cuArraysCopyToBatchWithOffset_kernel<<<gridSize,blockSize, 0 , stream>>> (
        chunk, windows, offsetH, offsetW, op);
    checkCudaAsyncErrors();
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to extract chips from a raw secondary image with varying offsets
 * @param chunk Input image as a large chunk
 * @param windows Output images as a batch of chips
 * @param offsetH (varying) offsets along height to extract chips
 * @param offsetW (varying) offsets along width to extract chips
 * @param stream cudaStream
 */
void cuArraysCopyToBatchWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    cudaStream_t stream)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW, ToComplex(), stream);
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 */
void cuArraysCopyToBatchAbsWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    cudaStream_t stream)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW, ToAmplitude(), stream);
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to load real images
 */
void cuArraysCopyToBatchWithOffsetR2C(ArraysView<const image_real_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    cudaStream_t stream)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW, ToComplex(), stream);
}

/**
 * Kernel for cuArraysCopyExtract with varying offsets: one thread per output pixel (outx, outy) of
 * the window blockIdx.z, copied from the input window at the offset offsets[blockIdx.z]
 */
template<typename T>
__global__ void cuArraysCopyExtractVaryingOffset(ArraysView<const T> imagesIn,
    ArraysView<T> imagesOut, ArraysView<const int2> offsets)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < imagesOut.height() && outy < imagesOut.width())
    {
        int idxImage = blockIdx.z;
        imagesOut(idxImage, outx, outy) =
            imagesIn(idxImage, outx + offsets[idxImage].x, outy + offsets[idxImage].y);
    }
}

/**
 * Copy a tile of images to another image, with starting pixels offsets
 * @param[in] imagesIn input images of dimension nImages*inNX*inNY
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] offsets,varying offsets for extraction
 */
template<typename T>
static void copyExtract(ArraysView<const T> imagesIn, ArraysView<T> imagesOut,
    ArraysView<const int2> offsets, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(divUp(imagesOut.height(),nthreads), divUp(imagesOut.width(),nthreads), imagesOut.count());
    cuArraysCopyExtractVaryingOffset<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn,
        imagesOut, offsets);
    checkCudaAsyncErrors();
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the varying pixels offsets (real arrays)
 */
void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> offsets, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offsets, stream);
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the varying pixels offsets (complex arrays)
 */
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    ArraysView<const int2> offsets, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offsets, stream);
}

/**
 * Kernel for cuArraysCopyExtractCorr: one thread per output pixel (outx, outy) of the window
 * blockIdx.z, copied from the input window around the center maxloc[blockIdx.z]
 * @note correlation surface extraction (Minyan Zhong)
 */
__global__ void cuArraysCopyExtractVaryingOffsetCorr(ArraysView<const real_type> imagesIn,
    ArraysView<real_type> imagesOut, ArraysView<const int2> maxloc)
{

    // get the image index
    int idxImage = blockIdx.z;

    // One thread per out point. Find the coordinates within the current image.
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    // check whether thread is within output image range
    if (outx < imagesOut.height() && outy < imagesOut.width())
    {
        // Find the corresponding input.
        int inx = outx + maxloc[idxImage].x - imagesOut.height()/2;
        int iny = outy + maxloc[idxImage].y - imagesOut.width()/2;

        imagesOut(idxImage, outx, outy) = imagesIn(idxImage, inx, iny);
    }
}

/**
 * Copy a tile of images centered at the max locations to another image
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] maxloc the max locations (as centers)
 * @note assume the extracted tiles are within the input images
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> maxloc, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = 16;

    dim3 threadsperblock(nthreads, nthreads,1);

    dim3 blockspergrid(divUp(imagesOut.height(),nthreads), divUp(imagesOut.width(),nthreads), imagesOut.count());

    cuArraysCopyExtractVaryingOffsetCorr<<<blockspergrid, threadsperblock,0, stream>>>(
        imagesIn, imagesOut, maxloc);
    checkCudaAsyncErrors();
}

/**
 * Kernel for cuArraysCopyExtractCorr with a stride and a valid region: one thread per output pixel
 * (outx, outy) of the window blockIdx.z, around the center maxloc[blockIdx.z]; the pixels outside
 * [validStart, validEnd) are set to 0 and marked as invalid (0) in imagesValid
 */
__global__ void cuArraysCopyExtractVaryingOffsetCorr(ArraysView<const real_type> imagesIn,
    ArraysView<real_type> imagesOut, ArraysView<int> imagesValid, ArraysView<const int2> maxloc,
    const int stride, const int2 validStart, const int2 validEnd)
{

    // get the image index
    int idxImage = blockIdx.z;

    // One thread per out point. Find the coordinates within the current image.
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    // check whether thread is within output image range
    if (outx < imagesOut.height() && outy < imagesOut.width())
    {
        // Find the corresponding input.
        int inx = maxloc[idxImage].x + (outx - imagesOut.height()/2)*stride;
        int iny = maxloc[idxImage].y + (outy - imagesOut.width()/2)*stride;

        // check whether inside of the valid region of the input image
        if (inx>=validStart.x && iny>=validStart.y && inx<validEnd.x && iny<validEnd.y)
        {
            // inside the boundary, copy over and mark the pixel as valid (1)
            imagesOut(idxImage, outx, outy) = imagesIn(idxImage, inx, iny);
            imagesValid(idxImage, outx, outy) = 1;
        }
        else {
            // outside, set it to 0 and mark the pixel as invalid (0)
            imagesOut(idxImage, outx, outy) = 0.0f;
            imagesValid(idxImage, outx, outy) = 0;
        }
    }
}

/**
 * Extract the surfaces around the max locations with a stride (e.g., a raw pixel spacing on an
 * oversampled surface), marking the pixels outside of a valid region of the input as invalid
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images, centered at the max locations
 * @param[out] imagesValid flags whether the pixels are within the valid region (1) or not (0)
 * @param[in] maxloc the max locations (as centers)
 * @param[in] stride the spacing of the extracted pixels in the input images
 * @param[in] validStart,validRange the valid region of the input images
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, int stride, int2 validStart,
    int2 validRange, cudaStream_t stream)
{
    const int nthreads = 16;

    dim3 threadsperblock(nthreads, nthreads,1);

    dim3 blockspergrid(divUp(imagesOut.height(),nthreads), divUp(imagesOut.width(),nthreads), imagesOut.count());

    const int2 validEnd = make_int2(validStart.x + validRange.x, validStart.y + validRange.y);
    cuArraysCopyExtractVaryingOffsetCorr<<<blockspergrid, threadsperblock,0, stream>>>(imagesIn,
        imagesOut, imagesValid, maxloc, stride, validStart, validEnd);
    checkCudaAsyncErrors();
}

/**
 * Copy a tile of images centered at the max locations to another image, accounting for boundary
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[out] imagesValid flags whether the pixels are within the input images (1) or not (0)
 * @param[in] maxloc the max locations (as centers)
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, cudaStream_t stream)
{
    cuArraysCopyExtractCorr(imagesIn, imagesOut, imagesValid, maxloc, 1,
        make_int2(0, 0), make_int2(imagesIn.height(), imagesIn.width()), stream);
}

// end of correlation surface extraction (Minyan Zhong)

/**
 * The element of the input as an element of the output (of the same type)
 */
template<typename T>
__device__ inline T extractConvert(const T &v) { return v; }
/**
 * The element of the input as an element of the output (from complex to real: the real part)
 */
__device__ inline real_type extractConvert(const complex_type &v) { return v.x; }

/**
 * Kernel for cuArraysCopyExtract with a fixed offset: one thread per output pixel (outx, outy) of
 * the window blockIdx.z, copied (converted with extractConvert) from the input window at the pixel
 * (outx + offsetX, outy + offsetY)
 */
template<typename T_in, typename T_out>
__global__ void cuArraysCopyExtractFixedOffset(ArraysView<const T_in> imagesIn,
    ArraysView<T_out> imagesOut, const int offsetX, const int offsetY)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < imagesOut.height() && outy < imagesOut.width())
    {
        const T_in &v = imagesIn(blockIdx.z, outx + offsetX, outy + offsetY);
        if constexpr (std::is_same_v<T_in, T_out>)
            imagesOut(blockIdx.z, outx, outy) = v;
        else
            imagesOut(blockIdx.z, outx, outy) = extractConvert(v);
    }
}

/**
 * Copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY)
 * @note for complex to real, the real part is taken
 */
template<typename T_in, typename T_out>
static void copyExtract(ArraysView<const T_in> imagesIn, ArraysView<T_out> imagesOut, int2 offset,
    cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = NTHREADS2D;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(divUp(imagesOut.height(),nthreads), divUp(imagesOut.width(),nthreads), imagesOut.count());
    cuArraysCopyExtractFixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn, imagesOut, offset.x, offset.y);
    checkCudaAsyncErrors();
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the fixed pixel offset (real arrays)
 */
void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offset, stream);
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the fixed pixel offset (the real parts of complex arrays)
 */
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offset, stream);
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the fixed pixel offset (complex arrays)
 */
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    int2 offset, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offset, stream);
}

/**
 * Extract the windows of imagesOut from imagesIn, starting at the fixed pixel offset (arrays of real3_type)
 */
void cuArraysCopyExtract(ArraysView<const real3_type> imagesIn, ArraysView<real3_type> imagesOut,
    int2 offset, cudaStream_t stream)
{
    copyExtract(imagesIn, imagesOut, offset, stream);
}

/**
 * Kernel for cuArraysCopyExtractAbs: one thread per output pixel (outx, outy) of the window
 * blockIdx.z, the amplitude of the input pixel (outx + offsetX, outy + offsetY)
 */
__global__ void cuArraysCopyExtractAbsFixedOffset(ArraysView<const complex_type> imagesIn,
    ArraysView<real_type> imagesOut, const int offsetX, const int offsetY)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < imagesOut.height() && outy < imagesOut.width())
    {
        imagesOut(blockIdx.z, outx, outy) =
            complexAbs(imagesIn(blockIdx.z, outx + offsetX, outy + offsetY));
    }
}
/**
 * Copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY), take amplitude
 */
void cuArraysCopyExtractAbs(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, cudaStream_t stream)
{
    //assert(imagesIn->height >= imagesOut && inNY >= outNY);
    const int nthreads = NTHREADS2D;
    dim3 threadsperblock(nthreads, nthreads,1);
    dim3 blockspergrid(divUp(imagesOut.height(),nthreads), divUp(imagesOut.width(),nthreads), imagesOut.count());
    cuArraysCopyExtractAbsFixedOffset<<<blockspergrid, threadsperblock,0, stream>>>
        (imagesIn, imagesOut, offset.x, offset.y);
    checkCudaAsyncErrors();
}

/**
 * Kernel for cuArraysCopyInsert: one thread per pixel (inx, iny) of the (first) array imageIn,
 * copied to the pixel (inx + offsetX, iny + offsetY) of the (first) array imageOut
 */
template<typename T>
__global__ void cuArraysCopyInsert_kernel(ArraysView<const T> imageIn, ArraysView<T> imageOut,
    const int offsetX, const int offsetY)
{
    int inx = threadIdx.x + blockDim.x*blockIdx.x;
    int iny = threadIdx.y + blockDim.y*blockIdx.y;
    if(inx < imageIn.height() && iny < imageIn.width())
        imageOut(inx+offsetX, iny+offsetY) = imageIn(inx, iny);
}

/**
 * Copy/insert images from a smaller size to a larger size from the location (offsetX, offsetY)
 */
template<typename T>
static void copyInsert(ArraysView<const T> imageIn, ArraysView<T> imageOut, int offsetX,
    int offsetY, cudaStream_t stream)
{
    const int nthreads = 16;
    dim3 threadsperblock(nthreads, nthreads);
    dim3 blockspergrid(divUp(imageIn.height(),nthreads), divUp(imageIn.width(),nthreads));
    cuArraysCopyInsert_kernel<<<blockspergrid, threadsperblock,0, stream>>>(imageIn, imageOut,
        offsetX, offsetY);
    checkCudaAsyncErrors();
}

/**
 * Insert the (first) array imageIn into the (first) array imageOut at the pixel (offsetX, offsetY) (complex arrays)
 */
void cuArraysCopyInsert(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    int offsetX, int offsetY, cudaStream_t stream)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY, stream);
}

/**
 * Insert the (first) array imageIn into the (first) array imageOut at the pixel (offsetX, offsetY) (arrays of real3_type)
 */
void cuArraysCopyInsert(ArraysView<const real3_type> imageIn, ArraysView<real3_type> imageOut,
    int offsetX, int offsetY, cudaStream_t stream)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY, stream);
}

/**
 * Insert the (first) array imageIn into the (first) array imageOut at the pixel (offsetX, offsetY) (real arrays)
 */
void cuArraysCopyInsert(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    int offsetX, int offsetY, cudaStream_t stream)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY, stream);
}

/**
 * Insert the (first) array imageIn into the (first) array imageOut at the pixel (offsetX, offsetY) (integer arrays)
 */
void cuArraysCopyInsert(ArraysView<const int> imageIn, ArraysView<int> imageOut,
    int offsetX, int offsetY, cudaStream_t stream)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY, stream);
}

/**
 * Kernel for cuArraysCopyPadded: one thread per output pixel (outx, outy) of the window blockIdx.z,
 * copied (converted to T_out) from the input window, or 0 outside of it
 */
template<typename T_in, typename T_out>
__global__ void cuArraysCopyPadded_kernel(ArraysView<const T_in> imageIn, ArraysView<T_out> imageOut)
{
    int outx = threadIdx.x + blockDim.x*blockIdx.x;
    int outy = threadIdx.y + blockDim.y*blockIdx.y;

    if(outx < imageOut.height() && outy < imageOut.width())
    {
        int idxImage = blockIdx.z;
        if(outx < imageIn.height() && outy < imageIn.width())
            imageOut(idxImage, outx, outy) = T_out{imageIn(idxImage, outx, outy)};
        else
            imageOut(idxImage, outx, outy) = T_out{0};
    }
}

/**
 * Copy images from a smaller size to a larger size while padding 0 for extra elements
 */
template<typename T_in, typename T_out>
static void copyPadded(ArraysView<const T_in> imageIn, ArraysView<T_out> imageOut, cudaStream_t stream)
{
    const int nthreads = 16;
    int nImages = imageIn.count();
    dim3 blockSize(nthreads, nthreads,1);
    dim3 gridSize(divUp(imageOut.height(),nthreads), divUp(imageOut.width(),nthreads), nImages);
    cuArraysCopyPadded_kernel<<<gridSize, blockSize, 0, stream>>>(imageIn, imageOut);
    checkCudaAsyncErrors();
}

/**
 * Copy the arrays imageIn to the (larger) arrays imageOut, padded with zeros (real arrays)
 */
void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    cudaStream_t stream)
{
    copyPadded(imageIn, imageOut, stream);
}

/**
 * Copy the arrays imageIn to the (larger) arrays imageOut, padded with zeros (from real to complex arrays)
 */
void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<complex_type> imageOut,
    cudaStream_t stream)
{
    copyPadded(imageIn, imageOut, stream);
}

/**
 * Copy the arrays imageIn to the (larger) arrays imageOut, padded with zeros (complex arrays)
 */
void cuArraysCopyPadded(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    cudaStream_t stream)
{
    copyPadded(imageIn, imageOut, stream);
}

/**
 * Kernel for cuArraysSetConstant: one thread per element (of all arrays), set to value
 */
__global__ void cuArraysSetConstant_kernel(ArraysView<real_type> image, int size, real_type value)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if(idx < size)
    {
        image[idx] = value;
    }
}

/**
 * Set real images to a constant value
 * @note use setZero if value=0 because cudaMemset is faster
 */
void cuArraysSetConstant(ArraysView<real_type> imageIn, real_type value, cudaStream_t stream)
{
    const int nthreads = 256;
    int size = imageIn.size();

    cuArraysSetConstant_kernel<<<divUp(size, nthreads), nthreads, 0, stream>>>
        (imageIn, size, value);
     checkCudaAsyncErrors();
}

/**
 * Kernel for cuArraysR2C: one thread per element (of all arrays), converted from real_type to
 * complex_type with a zero imaginary part
 */
__global__ void cuArraysR2C_kernel(ArraysView<const real_type> image1, ArraysView<complex_type> image2,
    int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx].x = image1[idx];
        image2[idx].y =  0.0;
    }
}

/**
 * Convert real images to complex images (set imaginary parts to 0)
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysR2C(ArraysView<const real_type> image1, ArraysView<complex_type> image2,
    cudaStream_t stream)
{
    int size = image1.size();
    cuArraysR2C_kernel<<<divUp(size, NTHREADS), NTHREADS, 0, stream>>>(image1, image2, size);
    checkCudaAsyncErrors();
}

/**
 * Kernel for cuArraysC2R: one thread per element (of all arrays), the real part of complex_type
 * as real_type
 */
__global__ void cuArraysC2R_kernel(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx] = image1[idx].x;
    }
}

/**
 * Take real part of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysC2R(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    cudaStream_t stream)
{
    int size = image1.size();
    cuArraysC2R_kernel<<<divUp(size, NTHREADS), NTHREADS, 0, stream>>>(image1, image2, size);
    checkCudaAsyncErrors();
}

/**
 * Kernel for cuArraysAbs: one thread per element (of all arrays), its amplitude
 */
__global__ void cuArraysAbs_kernel(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    int size)
{
    int idx =  threadIdx.x + blockDim.x*blockIdx.x;
    if(idx < size)
    {
        image2[idx] = complexAbs(image1[idx]);
    }
}

/**
 * Obtain abs (amplitudes) of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysAbs(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    cudaStream_t stream)
{
    int size = image1.size();
    cuArraysAbs_kernel<<<divUp(size, NTHREADS), NTHREADS, 0, stream>>>(image1, image2, size);
    checkCudaAsyncErrors();
}

// end of file

} // namespace
