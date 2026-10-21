/**
 * @file  Deramp.cpp
 * @brief Derampling a batch of 2D complex images (CPU backend)
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

#include "Deramp.h"
#include "Options.h"
#include "float2.h"
#include "logging.h"
#include "debug.h"

namespace pycuampcor::cpu {

/**
 * The phase ramp of a complex signal: the average phase differences between neighboring pixels,
 * atan(\sum imag / \sum real) of the products of the pixels with their conjugate neighbors
 * @param image the complex signal
 * @param axis 0: along x (down) only; 1: along y (across) only; others: along both
 * @return the phase differences along x and y (0 along an axis not estimated)
 */
static std::pair<double, double> estimateRamp(ArraysView<const complex_type> image, const int axis)
{
    const int imageNX = image.height();
    const int imageNY = image.width();

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
    return {phaseX, phaseY};
}

/**
 * Remove a phase ramp from a complex signal: multiply the pixel (ix, iy) by
 * exp(i (ix phaseX + iy phaseY))
 */
static void applyRamp(ArraysView<complex_type> image, double phaseX, double phaseY)
{
    const int imageNX = image.height();
    const int imageNY = image.width();
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

/**
 * Deramp complex signals (the method DerampMethod::Complex): the phase ramp of each signal
 * (estimateRamp) is removed (applyRamp)
 * @param[in,out] images input/output complex signals
 * @param[in] axis 0: deramp along x-axis; 1: deramp along y-axis; others: deramp along both axes
 */
static void cuLinearDeramp(ArraysView<complex_type> images, const int axis, stream_t)
{
    for(int idxImage = 0; idxImage < images.count(); idxImage++) {
        const ArraysView<complex_type> image = images.array(idxImage);
        const auto [phaseX, phaseY] = estimateRamp(image, axis);

#ifdef CUAMPCOR_DEBUG
        // output the phase ramp in debug mode
        logDebug("debug linear deramp az: ", phaseX, " rg: ", phaseY);
#endif
        applyRamp(image, phaseX, phaseY);
    }
}

/**
 * Remove the phase ramps of the complex windows (in place), along the axis, with the method
 * @note only DerampMethod::Complex deramps (with cuLinearDeramp); the other methods do nothing
 */
void cuDeramp(DerampMethod method, ArraysView<complex_type> images, DerampAxis axis, stream_t stream)
{
    // the axis values (down 0, across 1, both 2) are those of cuLinearDeramp
    if (method == DerampMethod::Complex)
        cuLinearDeramp(images, static_cast<int>(axis), stream);
}

/**
 * Remove the phase ramps of the reference windows from both the reference windows and the
 * secondary windows (in place), along the axis, with the method: the ramp of each reference window
 * is removed from it and from its secondary window, which keeps their relative phase
 * @note only DerampMethod::Complex deramps; the other methods do nothing
 */
void cuDerampByReference(DerampMethod method, ArraysView<complex_type> reference,
    ArraysView<complex_type> secondary, DerampAxis axis, stream_t)
{
    if (method != DerampMethod::Complex)
        return;
    for(int idxImage = 0; idxImage < reference.count(); idxImage++) {
        const auto [phaseX, phaseY] = estimateRamp(reference.array(idxImage), static_cast<int>(axis));
        applyRamp(reference.array(idxImage), phaseX, phaseY);
        applyRamp(secondary.array(idxImage), phaseX, phaseY);
    }
}

} // namespace
// end of file
