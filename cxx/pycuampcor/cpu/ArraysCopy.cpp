/**
 * @file ArraysCopy.cpp
 * @brief Utilities for copying/converting images to different format (CPU backend)
 *
 * All methods are declared in ArraysCopy.h
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
 *   see also ArraysPadding.cpp for other zero-padding utilities
 * cuArraysR2C cuArraysC2R cuArraysAbs to convert between different data types
 */

// dependencies
#include "ArraysCopy.h"
#include "float2.h"

namespace pycuampcor::cpu {

// convert image data to the internal complex type
static inline complex_type toComplex(const image_complex_type &v) { return make_complex_type(v.x, v.y); }
static inline complex_type toComplex(const image_real_type &v) { return make_complex_type(v, 0); }

// copy a chunk into a batch of chips with varying offsets, filling 0 for pixels outside the chunk
template<typename T_in, typename Op>
static void copyToBatchWithOffset(ArraysView<const T_in> chunk, ArraysView<complex_type> windows,
    ArraysView<const int> offsetX, ArraysView<const int> offsetY, Op op)
{
    const int inNX = chunk.height(), inNY = chunk.width();
    const int outNX = windows.height(), outNY = windows.width();
    for(int idxImage = 0; idxImage < windows.count(); idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int inx = offsetX[idxImage] + outx;
                int iny = offsetY[idxImage] + outy;
                if(inx>=0 && inx<inNX && iny>=0 && iny<inNY)
                    windows(idxImage, outx, outy) = op(chunk(inx, iny));
                else
                    windows(idxImage, outx, outy) = make_complex_type(0, 0);
            }
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to extract chips from a raw secondary image with varying offsets
 * @param chunk Input image as a large chunk
 * @param windows Output images as a batch of chips
 * @param offsetH (varying) offsets along height to extract chips
 * @param offsetW (varying) offsets along width to extract chips
 */
void cuArraysCopyToBatchWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW,
        [](const image_complex_type &v) { return toComplex(v); });
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 */
void cuArraysCopyToBatchAbsWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW,
        [](const image_complex_type &v) { return make_complex_type(complexAbs(v), 0); });
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to load real images
 */
void cuArraysCopyToBatchWithOffsetR2C(ArraysView<const image_real_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t)
{
    copyToBatchWithOffset(chunk, windows, offsetH, offsetW,
        [](const image_real_type &v) { return toComplex(v); });
}

/**
 * Copy a tile of images to another image, with starting pixels offsets
 * @param[in] imagesIn input images of dimension nImages*inNX*inNY
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] offsets, varying offsets for extraction
 */
template<typename T>
static void copyExtract(ArraysView<const T> imagesIn, ArraysView<T> imagesOut,
    ArraysView<const int2> offsets)
{
    const int outNX = imagesOut.height(), outNY = imagesOut.width();
    for(int idxImage = 0; idxImage < imagesOut.count(); idxImage++) {
        const int2 offset = offsets[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++)
                imagesOut(idxImage, outx, outy) = imagesIn(idxImage, outx + offset.x, outy + offset.y);
    }
}

void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> offsets, stream_t)
{
    copyExtract(imagesIn, imagesOut, offsets);
}

void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    ArraysView<const int2> offsets, stream_t)
{
    copyExtract(imagesIn, imagesOut, offsets);
}

/**
 * copy a tile of images centered at the max locations to another image
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] maxloc the max locations (as centers)
 * @note assume the extracted tiles are within the input images
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> maxloc, stream_t)
{
    const int outNX = imagesOut.height(), outNY = imagesOut.width();
    for(int idxImage = 0; idxImage < imagesOut.count(); idxImage++) {
        const int2 loc = maxloc[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int inx = outx + loc.x - outNX/2;
                int iny = outy + loc.y - outNY/2;
                imagesOut(idxImage, outx, outy) = imagesIn(idxImage, inx, iny);
            }
    }
}

/**
 * copy a tile of images centered at the max locations to another image, accounting for boundary
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[out] imagesValid flags whether the pixels are within the input images (1) or not (0)
 * @param[in] maxloc the max locations (as centers)
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, stream_t stream)
{
    cuArraysCopyExtractCorr(imagesIn, imagesOut, imagesValid, maxloc, 1,
        make_int2(0, 0), make_int2(imagesIn.height(), imagesIn.width()), stream);
}

/**
 * extract the surfaces around the max locations with a stride (e.g., a raw pixel spacing on an
 * oversampled surface), marking the pixels outside of a valid region of the input as invalid
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images, centered at the max locations
 * @param[out] imagesValid flags whether the pixels are within the valid region (1) or not (0)
 * @param[in] maxloc the max locations (as centers)
 * @param[in] stride the spacing of the extracted pixels in the input images
 * @param[in] validStart, validRange the valid region of the input images
 */
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, int stride, int2 validStart,
    int2 validRange, stream_t)
{
    const int outNX = imagesOut.height(), outNY = imagesOut.width();
    const int validEndX = validStart.x + validRange.x, validEndY = validStart.y + validRange.y;
    for(int idxImage = 0; idxImage < imagesOut.count(); idxImage++) {
        const int2 loc = maxloc[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int inx = loc.x + (outx - outNX/2)*stride;
                int iny = loc.y + (outy - outNY/2)*stride;
                if (inx>=validStart.x && iny>=validStart.y && inx<validEndX && iny<validEndY) {
                    // inside the boundary, copy over and mark the pixel as valid (1)
                    imagesOut(idxImage, outx, outy) = imagesIn(idxImage, inx, iny);
                    imagesValid(idxImage, outx, outy) = 1;
                }
                else {
                    // outside, set it to 0 and mark the pixel as invalid (0)
                    imagesOut(idxImage, outx, outy) = 0;
                    imagesValid(idxImage, outx, outy) = 0;
                }
            }
    }
}

// element conversion for cuArraysCopyExtract with a fixed offset
template<typename T_in, typename T_out> struct ExtractConverter {
    static T_out apply(const T_in &v) { return v; }
};
template<> struct ExtractConverter<complex_type, real_type> {
    static real_type apply(const complex_type &v) { return v.x; }
};

/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY)
 * @note for complex to real, the real part is taken
 */
template<typename T_in, typename T_out>
static void copyExtract(ArraysView<const T_in> imagesIn, ArraysView<T_out> imagesOut, int2 offset)
{
    const int outNX = imagesOut.height(), outNY = imagesOut.width();
    for(int idxImage = 0; idxImage < imagesOut.count(); idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++)
                imagesOut(idxImage, outx, outy) = ExtractConverter<T_in, T_out>::apply(
                    imagesIn(idxImage, outx + offset.x, outy + offset.y));
}

void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t)
{
    copyExtract(imagesIn, imagesOut, offset);
}

void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t)
{
    copyExtract(imagesIn, imagesOut, offset);
}

void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    int2 offset, stream_t)
{
    copyExtract(imagesIn, imagesOut, offset);
}

void cuArraysCopyExtract(ArraysView<const real3_type> imagesIn, ArraysView<real3_type> imagesOut,
    int2 offset, stream_t)
{
    copyExtract(imagesIn, imagesOut, offset);
}

/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY), take amplitude
 */
void cuArraysCopyExtractAbs(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t)
{
    const int outNX = imagesOut.height(), outNY = imagesOut.width();
    for(int idxImage = 0; idxImage < imagesOut.count(); idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++)
                imagesOut(idxImage, outx, outy) =
                    complexAbs(imagesIn(idxImage, outx + offset.x, outy + offset.y));
}

/**
 * copy/insert images from a smaller size to a larger size from the location (offsetX, offsetY)
 */
template<typename T>
static void copyInsert(ArraysView<const T> imageIn, ArraysView<T> imageOut, int offsetX, int offsetY)
{
    for(int inx = 0; inx < imageIn.height(); inx++)
        for(int iny = 0; iny < imageIn.width(); iny++)
            imageOut(inx+offsetX, iny+offsetY) = imageIn(inx, iny);
}

void cuArraysCopyInsert(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    int offsetX, int offsetY, stream_t)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY);
}

void cuArraysCopyInsert(ArraysView<const real3_type> imageIn, ArraysView<real3_type> imageOut,
    int offsetX, int offsetY, stream_t)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY);
}

void cuArraysCopyInsert(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    int offsetX, int offsetY, stream_t)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY);
}

void cuArraysCopyInsert(ArraysView<const int> imageIn, ArraysView<int> imageOut,
    int offsetX, int offsetY, stream_t)
{
    copyInsert(imageIn, imageOut, offsetX, offsetY);
}

// element conversion for cuArraysCopyPadded
template<typename T_in, typename T_out> struct PaddedConverter {
    static T_out apply(const T_in &v) { return v; }
};
template<> struct PaddedConverter<real_type, complex_type> {
    static complex_type apply(const real_type &v) { return make_complex_type(v, 0); }
};

/**
 * copy images from a smaller size to a larger size while padding 0 for extra elements
 */
template<typename T_in, typename T_out>
static void copyPadded(ArraysView<const T_in> imageIn, ArraysView<T_out> imageOut)
{
    const int inNX = imageIn.height(), inNY = imageIn.width();
    const int outNX = imageOut.height(), outNY = imageOut.width();
    for(int idxImage = 0; idxImage < imageIn.count(); idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                if(outx < inNX && outy < inNY)
                    imageOut(idxImage, outx, outy) = PaddedConverter<T_in, T_out>::apply(
                        imageIn(idxImage, outx, outy));
                else
                    imageOut(idxImage, outx, outy) = T_out{0};
            }
}

void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    stream_t)
{
    copyPadded(imageIn, imageOut);
}

void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<complex_type> imageOut,
    stream_t)
{
    copyPadded(imageIn, imageOut);
}

void cuArraysCopyPadded(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    stream_t)
{
    copyPadded(imageIn, imageOut);
}

/**
 * Set real images to a constant value
 */
void cuArraysSetConstant(ArraysView<real_type> imageIn, real_type value, stream_t)
{
    const size_t size = imageIn.size();
    for(size_t idx = 0; idx < size; idx++)
        imageIn.data_handle()[idx] = value;
}

/**
 * Convert real images to complex images (set imaginary parts to 0)
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysR2C(ArraysView<const real_type> image1, ArraysView<complex_type> image2, stream_t)
{
    const size_t size = image1.size();
    for(size_t idx = 0; idx < size; idx++)
        image2.data_handle()[idx] = make_complex_type(image1.data_handle()[idx], 0);
}

/**
 * Take real part of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysC2R(ArraysView<const complex_type> image1, ArraysView<real_type> image2, stream_t)
{
    const size_t size = image1.size();
    for(size_t idx = 0; idx < size; idx++)
        image2.data_handle()[idx] = image1.data_handle()[idx].x;
}

/**
 * Obtain abs (amplitudes) of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysAbs(ArraysView<const complex_type> image1, ArraysView<real_type> image2, stream_t)
{
    const size_t size = image1.size();
    for(size_t idx = 0; idx < size; idx++)
        image2.data_handle()[idx] = complexAbs(image1.data_handle()[idx]);
}

} // namespace
// end of file
