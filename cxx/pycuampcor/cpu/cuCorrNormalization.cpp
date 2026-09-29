/*
 * @file cuCorrNormalization.cpp
 * @brief Utilities to normalize the correlation surface (CPU backend)
 *
 * The normalization of the correlation surface itself is done with the
 * sum area tables, see cuCorrNormalizationSAT.cpp
 */

#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

/**
 * Compute and subtract mean values from images
 * @param[in,out] images Input/Output images
 */
void cuArraysSubtractMean(cuArrays<real_type> *images, stream_t)
{
    const int imageSize = images->size;
    const real_type invSize = real_type(1)/imageSize;
    for(int idxImage = 0; idxImage < images->count; idxImage++) {
        real_type *image = images->devData + (size_t)idxImage*imageSize;
        double sum = 0.0;
        for(int i = 0; i < imageSize; i++)
            sum += image[i];
        const real_type mean = sum * invSize;
        for(int i = 0; i < imageSize; i++)
            image[i] -= mean;
    }
}

/**
 * Compute the sum of squares of images (for SNR)
 * @param[in] images Input images
 * @param[out] imagesSum sum of squares
 */
void cuArraysSumSquare(cuArrays<real_type> *images, cuArrays<real_type> *imagesSum, stream_t)
{
    const int imageSize = images->size;
    for(int idxImage = 0; idxImage < images->count; idxImage++) {
        const real_type *image = images->devData + (size_t)idxImage*imageSize;
        double sum = 0.0;
        for(int i = 0; i < imageSize; i++)
            sum += image[i]*image[i];
        imagesSum->devData[idxImage] = sum;
    }
}

/**
 * Compute the sum of squares of images and the count of valid pixels (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum sum of squares
 * @param[out] imagesValidCount count of total valid pixels
 */
void cuArraysSumCorr(cuArrays<real_type> *images, cuArrays<int> *imagesValid, cuArrays<real_type> *imagesSum,
    cuArrays<int> *imagesValidCount, stream_t)
{
    const int imageSize = images->size;
    for(int idxImage = 0; idxImage < images->count; idxImage++) {
        const real_type *image = images->devData + (size_t)idxImage*imageSize;
        const int *imageValid = imagesValid->devData + (size_t)idxImage*imageSize;
        double sum = 0.0;
        int count = 0;
        for(int i = 0; i < imageSize; i++) {
            sum += image[i]*image[i];
            count += imageValid[i];
        }
        imagesSum->devData[idxImage] = sum;
        imagesValidCount->devData[idxImage] = count;
    }
}

} // namespace
// end of file
