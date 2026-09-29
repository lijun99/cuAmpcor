/*
 * @file  cuDeramp.cpp
 * @brief Derampling a batch of 2D complex images (CPU backend)
 *
 * A phase ramp is equivalent to a frequency shift in frequency domain,
 *   which needs to be removed (deramping) in order to move the band center
 *   to zero. This is necessary before oversampling a complex signal.
 * Method 1: each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 *   The average is weighted by the amplitudes (coherence).
 * Method 0 or else: skip deramping
 *
 */

#include "cuAmpcorUtil.h"
#include <cmath>
#include <cstdio>

namespace pycuampcor::cpu {

/**
 * Deramp a complex signal with Method 1
 * @brief Each signal is decomposed into real and imaginary parts,
 *   and the average phase shift is obtained as atan(\sum imag / \sum real).
 * @param[in,out] images input/output complex signals
 * @param[in] axis 0: deramp along x-axis; 1: deramp along y-axis; others: deramp along both axes
 */
void cuLinearDeramp(cuArrays<complex_type> *images, const int axis, stream_t)
{
    const int imageNX = images->height;
    const int imageNY = images->width;
    const int imageSize = images->size;

    for(int idxImage = 0; idxImage < images->count; idxImage++) {
        complex_type *image = images->devData + (size_t)idxImage*imageSize;

        // compute phase difference along y direction
        double phaseY = 0.0;
        if (axis != 0) {
            double sumx = 0.0, sumy = 0.0;
            for(int ix = 0; ix < imageNX; ix++)
                for(int iy = 0; iy < imageNY-1; iy++) {
                    int pixelIdx = ix*imageNY + iy;
                    complex_type cprod = complexMulConj(image[pixelIdx], image[pixelIdx+1]);
                    sumx += cprod.x;
                    sumy += cprod.y;
                }
            phaseY = std::atan2(sumy, sumx);
        }

        // compute phase difference along x direction
        double phaseX = 0.0;
        if (axis != 1) {
            double sumx = 0.0, sumy = 0.0;
            for(int pixelIdx = 0; pixelIdx < (imageNX-1)*imageNY; pixelIdx++) {
                complex_type cprod = complexMulConj(image[pixelIdx], image[pixelIdx+imageNY]);
                sumx += cprod.x;
                sumy += cprod.y;
            }
            phaseX = std::atan2(sumy, sumx);
        }

#ifdef CUAMPCOR_DEBUG
        // output the phase ramp in debug mode
        printf("debug linear deramp az: %g rg: %g\n", phaseX, phaseY);
#endif

        for(int ix = 0; ix < imageNX; ix++)
            for(int iy = 0; iy < imageNY; iy++) {
                int pixelIdx = ix*imageNY + iy;
                double phase = ix*phaseX + iy*phaseY;
                double phase_sin = std::sin(phase);
                double phase_cos = std::cos(phase);
                complex_type v = image[pixelIdx];
                image[pixelIdx] = make_complex_type(
                    v.x*phase_cos - v.y*phase_sin,
                    v.x*phase_sin + v.y*phase_cos);
            }
    }
}

void cuDeramp(int method, cuArrays<complex_type> *images, const int axis, stream_t stream)
{
    switch(method) {
    case 1:
        cuLinearDeramp(images, axis, stream);
        break;
    default:
        break;
    }
}

} // namespace
// end of file
