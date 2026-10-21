/**
 * @file  SincOverSampler.h
 * @brief A class performs sinc interpolation/oversampling (CPU backend)
 *
 * Oversample a given 2d signal by i_covs factor.
 * Only signals within(-i_sincwindow, i_sincwindow) are oversampled
 * The interpolation zone may also be shifted, if the max location is not at the center.
 *
 * The sinc interpolation is based on the formula
 *   $$x(t) = \sum_{n=-\infty}^{\infty} x_n f( \Omega_c t-n )$$
 *   with $f(x) = \text{sinc}(x)$, or a complex filter
 *   such as the sinc(x) convoluted with Hamming Window used here.
 *   In practice, a finite length of n (i_intplength) is used for interpolation.
 *
 * @note most parameters are currently hardwired; you need to change
 *    the source code below if you need to adjust the parameters.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

// dependencites

namespace pycuampcor::cpu {

constexpr float PI = 3.14159265359f;   ///< the constant pi

/// the sinc oversampler of real images (correlation surfaces), only within a window around
/// their centers (or shifted centers, e.g., at the peaks)
class cuSincOverSamplerR2R
{
 private:
    static const int i_sincwindow = 2;   ///< the oversampling is only performed within \pm i_sincwindow*i_covs around the peak
    static const int i_weight = 1;       ///< weight for cos() pedestal
    const real_type r_pedestal = 0.0;       ///< height of pedestal
    const real_type r_beta = 0.75f;          ///< a low-band pass
    const real_type r_relfiltlen = 6.0f;     ///< relative filter length

    static const int i_decfactor = 4096; ///< max decimals between original grid to set up the sinc kernel

    int i_covs;         ///< oversampling factor
    int i_intplength;   ///< actual filter length = r_relfiltlen/r_beta
    int i_filtercoef;   ///< i_intplength*i_decfactor (the sinc kernel has i_filtercoef+1 coefficients)

    std::vector<real_type> r_filter;   ///< the sinc kernel, of size i_filtercoef+1

    stream_t stream;   ///< the stream (not used)

 public:
    /// constructor, with the oversampling factor i_covs_
    cuSincOverSamplerR2R(const int i_covs_, stream_t stream_);
    /// set up the sinc interpolation coefficients (r_filter)
    void cuSetupSincKernel();
    /// oversample the images within the windows around their centers, shifted by
    /// center*oversamplingFactor (oversampled pixels, one shift per image); the rest of
    /// imagesOut is set to zero
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
        ArraysView<const int2> center, int oversamplingFactor);
    /// the same, with one shift (center) for all images
    void execute(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut, int2 center,
        int oversamplingFactor);

    /// destructor
    ~cuSincOverSamplerR2R();
};

} // namespace
// end of file
