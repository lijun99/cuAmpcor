/**
 * @file cuArraysCopy.cpp
 * @brief Utilities for copying/converting images to different format (CPU backend)
 *
 * All methods are declared in cuAmpcorUtil.h
 * cudaArraysCopyToBatch to extract a batch of windows from the raw image
 *   various implementations include:
 *   1. fixed or varying offsets, as start pixels for windows
 *   2. complex to complex, usually
 *   3. complex to (amplitude,0), for TOPS
 *   4. real to complex, for real images
 * cuArraysCopyExtract to extract(shrink in size) from a batch of windows to another batch
 *   overloaded for different data types
 * cuArraysCopyInsert to insert a batch of windows (smaller in size) to another batch
 *   overloaded for different data types
 * cuArraysCopyPadded to insert a batch of windows to another batch while padding 0s for rest elements
 *   used for fft oversampling
 *   see also cuArraysPadding.cpp for other zero-padding utilities
 * cuArraysR2C cuArraysC2R cuArraysAbs to convert between different data types
 */

// dependencies
#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

// convert image data to the internal complex type
static inline complex_type toComplex(const image_complex_type &v) { return make_complex_type(v.x, v.y); }
static inline complex_type toComplex(const image_real_type &v) { return make_complex_type(v, 0); }

/**
 * Copy a chunk into a batch of chips for a given stride
 * @note used to extract chips from a raw image
 * @param image1 Input image as a large chunk
 * @param image2 Output images as a batch of chips
 * @param strideH stride along height to extract chips
 * @param strideW stride along width to extract chips
 */
void cuArraysCopyToBatch(cuArrays<image_complex_type> *image1, cuArrays<complex_type> *image2,
    int strideH, int strideW, stream_t)
{
    const int inNY = image1->width;
    const int outNX = image2->height, outNY = image2->width;
    for(int idxImage = 0; idxImage < image2->count; idxImage++) {
        const int idxImageX = idxImage/image2->countW;
        const int idxImageY = idxImage%image2->countW;
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
                int idxIn = (idxImageX*strideH+outx)*inNY + idxImageY*strideW+outy;
                image2->devData[idxOut] = toComplex(image1->devData[idxIn]);
            }
    }
}

// copy a chunk into a batch of chips with varying offsets, filling 0 for pixels outside the chunk
template<typename T_in, typename Op>
static void copyToBatchWithOffset(const T_in *imageIn, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetX, const int *offsetY, Op op)
{
    const int outNX = image2->height, outNY = image2->width;
    for(int idxImage = 0; idxImage < image2->count; idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
                int inx = offsetX[idxImage] + outx;
                int iny = offsetY[idxImage] + outy;
                if(inx>=0 && inx<inNX && iny>=0 && iny<inNY)
                    image2->devData[idxOut] = op(imageIn[inx*inNY+iny]);
                else
                    image2->devData[idxOut] = make_complex_type(0, 0);
            }
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to extract chips from a raw secondary image with varying offsets
 * @param image1 Input image as a large chunk
 * @param inNX, inNY the size of the chunk
 * @param image2 Output images as a batch of chips
 * @param offsetH (varying) offsets along height to extract chips
 * @param offsetW (varying) offsets along width to extract chips
 */
void cuArraysCopyToBatchWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t)
{
    copyToBatchWithOffset(image1->devData, inNX, inNY, image2, offsetH, offsetW,
        [](const image_complex_type &v) { return toComplex(v); });
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note similar to cuArraysCopyToBatchWithOffset, but take amplitudes instead
 */
void cuArraysCopyToBatchAbsWithOffset(cuArrays<image_complex_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t)
{
    copyToBatchWithOffset(image1->devData, inNX, inNY, image2, offsetH, offsetW,
        [](const image_complex_type &v) { return make_complex_type(complexAbs(v), 0); });
}

/**
 * Copy a chunk into a batch of chips with varying offsets/strides
 * @note used to load real images
 */
void cuArraysCopyToBatchWithOffsetR2C(cuArrays<image_real_type> *image1, const int inNX, const int inNY,
    cuArrays<complex_type> *image2, const int *offsetH, const int* offsetW, stream_t)
{
    copyToBatchWithOffset(image1->devData, inNX, inNY, image2, offsetH, offsetW,
        [](const image_real_type &v) { return toComplex(v); });
}

/**
 * Copy a chunk into a batch of chips for a given stride, taking amplitudes (with the FFT factor)
 * @param image1 Input image as a large chunk
 * @param image2 Output images as a batch of chips
 * @param strideH offsets along height to extract chips
 * @param strideW offsets along width to extract chips
 */
void cuArraysCopyC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2,
    int strideH, int strideW, stream_t)
{
    const int inNY = image1->width;
    const int outNX = image2->height, outNY = image2->width;
    const real_type factor = real_type(1)/image1->size; //the FFT factor
    for(int idxImage = 0; idxImage < image2->count; idxImage++) {
        const int idxImageX = idxImage/image2->countW;
        const int idxImageY = idxImage%image2->countW;
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = idxImage*outNX*outNY + outx*outNY + outy;
                int idxIn = (idxImageX*strideH+outx)*inNY + idxImageY*strideW+outy;
                image2->devData[idxOut] = complexAbs(image1->devData[idxIn])*factor;
            }
    }
}

/**
 * Copy a tile of images to another image, with starting pixels offsets
 * @param[in] imagesIn input images of dimension nImages*inNX*inNY
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] offsets, varying offsets for extraction
 */
template<typename T>
void cuArraysCopyExtract(cuArrays<T> *imagesIn, cuArrays<T> *imagesOut, cuArrays<int2> *offsets, stream_t)
{
    const int inNX = imagesIn->height, inNY = imagesIn->width;
    const int outNX = imagesOut->height, outNY = imagesOut->width;
    for(int idxImage = 0; idxImage < imagesOut->count; idxImage++) {
        const int2 offset = offsets->devData[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = (idxImage*outNX + outx)*outNY+outy;
                int idxIn = (idxImage*inNX + outx + offset.x)*inNY + outy + offset.y;
                imagesOut->devData[idxOut] = imagesIn->devData[idxIn];
            }
    }
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<real_type> *in, cuArrays<real_type> *out, cuArrays<int2> *offsets, stream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<complex_type> *out, cuArrays<int2> *offsets, stream_t);

/**
 * copy a tile of images centered at the max locations to another image
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images of dimension nImages*outNX*outNY
 * @param[in] maxloc the max locations (as centers)
 * @note assume the extracted tiles are within the input images
 */
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int2> *maxloc, stream_t)
{
    const int inNX = imagesIn->height, inNY = imagesIn->width;
    const int outNX = imagesOut->height, outNY = imagesOut->width;
    for(int idxImage = 0; idxImage < imagesOut->count; idxImage++) {
        const int2 loc = maxloc->devData[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int inx = outx + loc.x - outNX/2;
                int iny = outy + loc.y - outNY/2;
                int idxOut = (idxImage * outNX + outx) * outNY + outy;
                int idxIn = (idxImage * inNX + inx) * inNY + iny;
                imagesOut->devData[idxOut] = imagesIn->devData[idxIn];
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
void cuArraysCopyExtractCorr(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut, cuArrays<int> *imagesValid, cuArrays<int2> *maxloc, stream_t)
{
    const int inNX = imagesIn->height, inNY = imagesIn->width;
    const int outNX = imagesOut->height, outNY = imagesOut->width;
    for(int idxImage = 0; idxImage < imagesOut->count; idxImage++) {
        const int2 loc = maxloc->devData[idxImage];
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int inx = outx + loc.x - outNX/2;
                int iny = outy + loc.y - outNY/2;
                int idxOut = (idxImage * outNX + outx) * outNY + outy;
                if (inx>=0 && iny>=0 && inx<inNX && iny<inNY) {
                    // inside the boundary, copy over and mark the pixel as valid (1)
                    imagesOut->devData[idxOut] = imagesIn->devData[(idxImage * inNX + inx) * inNY + iny];
                    imagesValid->devData[idxOut] = 1;
                }
                else {
                    // outside, set it to 0 and mark the pixel as invalid (0)
                    imagesOut->devData[idxOut] = 0;
                    imagesValid->devData[idxOut] = 0;
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
void cuArraysCopyExtract(cuArrays<T_in> *imagesIn, cuArrays<T_out> *imagesOut, int2 offset, stream_t)
{
    const int inNX = imagesIn->height, inNY = imagesIn->width;
    const int outNX = imagesOut->height, outNY = imagesOut->width;
    for(int idxImage = 0; idxImage < imagesOut->count; idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = (idxImage * outNX + outx)*outNY+outy;
                int idxIn = (idxImage*inNX + outx + offset.x)*inNY + outy + offset.y;
                imagesOut->devData[idxOut] = ExtractConverter<T_in, T_out>::apply(imagesIn->devData[idxIn]);
            }
}

// instantiate the above template for the data types we need
template void cuArraysCopyExtract(cuArrays<real_type> *in, cuArrays<real_type> *out, int2 offset, stream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<real_type> *out, int2 offset, stream_t);
template void cuArraysCopyExtract(cuArrays<complex_type> *in, cuArrays<complex_type> *out, int2 offset, stream_t);
template void cuArraysCopyExtract(cuArrays<real3_type> *in, cuArrays<real3_type> *out, int2 offset, stream_t);

/**
 * copy/extract images from a large size to
 * a smaller size from the location (offsetX, offsetY), take amplitude
 */
void cuArraysCopyExtractAbs(cuArrays<complex_type> *imagesIn, cuArrays<real_type> *imagesOut, int2 offset, stream_t)
{
    const int inNX = imagesIn->height, inNY = imagesIn->width;
    const int outNX = imagesOut->height, outNY = imagesOut->width;
    for(int idxImage = 0; idxImage < imagesOut->count; idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = (idxImage * outNX + outx)*outNY+outy;
                int idxIn = (idxImage*inNX + outx + offset.x)*inNY + outy + offset.y;
                imagesOut->devData[idxOut] = complexAbs(imagesIn->devData[idxIn]);
            }
}

/**
 * copy/insert images from a smaller size to a larger size from the location (offsetX, offsetY)
 */
template<typename T>
void cuArraysCopyInsert(cuArrays<T> *imageIn, cuArrays<T> *imageOut, int offsetX, int offsetY, stream_t)
{
    const int inNX = imageIn->height, inNY = imageIn->width;
    const int outNY = imageOut->width;
    for(int inx = 0; inx < inNX; inx++)
        for(int iny = 0; iny < inNY; iny++)
            imageOut->devData[IDX2R(inx+offsetX, iny+offsetY, outNY)] = imageIn->devData[IDX2R(inx, iny, inNY)];
}

// instantiate the above template for the data types we need
template void cuArraysCopyInsert(cuArrays<complex_type>* in, cuArrays<complex_type>* out, int offX, int offY, stream_t);
template void cuArraysCopyInsert(cuArrays<real3_type>* in, cuArrays<real3_type>* out, int offX, int offY, stream_t);
template void cuArraysCopyInsert(cuArrays<real_type>* in, cuArrays<real_type>* out, int offX, int offY, stream_t);
template void cuArraysCopyInsert(cuArrays<int>* in, cuArrays<int>* out, int offX, int offY, stream_t);

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
void cuArraysCopyPadded(cuArrays<T_in> *imageIn, cuArrays<T_out> *imageOut, stream_t)
{
    const int inNX = imageIn->height, inNY = imageIn->width;
    const int outNX = imageOut->height, outNY = imageOut->width;
    for(int idxImage = 0; idxImage < imageIn->count; idxImage++)
        for(int outx = 0; outx < outNX; outx++)
            for(int outy = 0; outy < outNY; outy++) {
                int idxOut = IDX2R(outx, outy, outNY)+idxImage*imageOut->size;
                if(outx < inNX && outy < inNY)
                    imageOut->devData[idxOut] = PaddedConverter<T_in, T_out>::apply(
                        imageIn->devData[IDX2R(outx, outy, inNY)+idxImage*imageIn->size]);
                else
                    imageOut->devData[idxOut] = T_out{0};
            }
}

// instantiate the above template for the data types we need
template void cuArraysCopyPadded(cuArrays<real_type> *imageIn, cuArrays<real_type> *imageOut, stream_t);
template void cuArraysCopyPadded(cuArrays<real_type> *imageIn, cuArrays<complex_type> *imageOut, stream_t);
template void cuArraysCopyPadded(cuArrays<complex_type> *imageIn, cuArrays<complex_type> *imageOut, stream_t);

/**
 * Set real images to a constant value
 */
void cuArraysSetConstant(cuArrays<real_type> *imageIn, real_type value, stream_t)
{
    const size_t size = imageIn->getSize();
    for(size_t idx = 0; idx < size; idx++)
        imageIn->devData[idx] = value;
}

/**
 * Convert real images to complex images (set imaginary parts to 0)
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysR2C(cuArrays<real_type> *image1, cuArrays<complex_type> *image2, stream_t)
{
    const size_t size = image1->getSize();
    for(size_t idx = 0; idx < size; idx++)
        image2->devData[idx] = make_complex_type(image1->devData[idx], 0);
}

/**
 * Take real part of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysC2R(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t)
{
    const size_t size = image1->getSize();
    for(size_t idx = 0; idx < size; idx++)
        image2->devData[idx] = image1->devData[idx].x;
}

/**
 * Obtain abs (amplitudes) of complex images
 * @param[in] image1 input images
 * @param[out] image2 output images
 */
void cuArraysAbs(cuArrays<complex_type> *image1, cuArrays<real_type> *image2, stream_t)
{
    const size_t size = image1->getSize();
    for(size_t idx = 0; idx < size; idx++)
        image2->devData[idx] = complexAbs(image1->devData[idx]);
}

} // namespace
// end of file
