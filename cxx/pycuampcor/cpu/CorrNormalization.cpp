/*
 * @file CorrNormalization.cpp
 * @brief Utilities to normalize the correlation surface (CPU backend)
 *
 * The normalization of the correlation surface itself is done with the
 * sum area tables, see CorrNormalizationSAT.cpp
 */

#include "CorrNormalization.h"
#include "float2.h"

namespace pycuampcor::cpu {

/**
 * Compute and subtract mean values from images
 * @param[in,out] images Input/Output images
 */
void cuArraysSubtractMean(ArraysView<real_type> images, stream_t)
{
    const int imageSize = images.arraySize();
    const real_type invSize = real_type(1)/imageSize;
    for(int idxImage = 0; idxImage < images.count(); idxImage++) {
        real_type *image = images.array(idxImage).data_handle();
        double sum = 0.0;
        for(int i = 0; i < imageSize; i++)
            sum += image[i];
        const real_type mean = sum * invSize;
        for(int i = 0; i < imageSize; i++)
            image[i] -= mean;
    }
}

/**
 * Compute the sum of squares of images and the count of valid pixels (for SNR)
 * @param[in] images Input images
 * @param[in] imagesValid validity flags for each pixel
 * @param[out] imagesSum sum of squares
 * @param[out] imagesValidCount count of total valid pixels
 */
void cuArraysSumCorr(ArraysView<const real_type> images, ArraysView<const int> imagesValid,
    ArraysView<real_type> imagesSum, ArraysView<int> imagesValidCount, stream_t)
{
    const int imageSize = images.arraySize();
    for(int idxImage = 0; idxImage < images.count(); idxImage++) {
        const real_type *image = images.array(idxImage).data_handle();
        const int *imageValid = imagesValid.array(idxImage).data_handle();
        double sum = 0.0;
        int count = 0;
        for(int i = 0; i < imageSize; i++) {
            sum += image[i]*image[i];
            count += imageValid[i];
        }
        imagesSum[idxImage] = sum;
        imagesValidCount[idxImage] = count;
    }
}

} // namespace
// end of file
