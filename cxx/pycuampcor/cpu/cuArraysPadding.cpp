/*
 * @file  cuArraysPadding.cpp
 * @brief Utilities for padding zeros to cuArrays for FFT (zero padding in the middle), CPU backend
 */

#include "cuAmpcorUtil.h"

namespace pycuampcor::cpu {

/**
 * Padding zeros for FFT oversampling
 * @param[in] image1 input images
 * @param[out] image2 output images
 * @note To keep the band center at (0,0), move quads to corners and pad zeros in the middle
 * @note for both even and odd sequences, siding Nyquist frequency for even length with negative frequency
 * for even N - positive f[0, ..., N/2-1],
 *              zeros 0...0,
 *              negative f[N/2], f[N/2+1, ..., N-1]
 * for odd N - positive f[0, ..., (N-1)/2],
 *             zeros 0...0,
 *             negative f [(N+1)/2, ..., N-1]
 */
void cuArraysFFTPaddingMany(cuArrays<complex_type> *image1, cuArrays<complex_type> *image2, stream_t stream)
{
    const int height1 = image1->height, width1 = image1->width, size1 = image1->size;
    const int height2 = image2->height, width2 = image2->width, size2 = image2->size;
    const real_type factor = real_type(1)/size1;

    image2->setZero(stream);
    for(int imageIdx = 0; imageIdx < image1->count; imageIdx++)
        for(int x1 = 0; x1 < height1; x1++) {
            // determine the quadrants
            // divup the length to be consistent with both even and odd lengths
            const int x2 = (x1 < (height1+1)/2) ? x1 : height2 - height1 + x1;
            for(int y1 = 0; y1 < width1; y1++) {
                const int y2 = (y1 < (width1+1)/2) ? y1 : width2 - width1 + y1;
                image2->devData[IDX2R(x2, y2, width2)+imageIdx*size2]
                    = image1->devData[IDX2R(x1, y1, width1)+imageIdx*size1]*factor;
            }
        }
}

} // namespace
// end of file
