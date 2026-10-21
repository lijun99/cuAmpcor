/**
 * @file  CorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CPU backend)
 *
 * The kernel launcher cuCorrTimeDomain, and the correlator Correlator<Algorithm::Time, real_type,
 * Device::CPU> on it. Implemented in CorrTimeDomain.cpp.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"

namespace pycuampcor::cpu {

/// cross-correlate the templates (reference windows) with the images (secondary windows) in the
/// time domain: the k-th surface of results, of (image - template + 1) pixels each way, from the k-th
/// template and image
void cuCorrTimeDomain(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results, stream_t stream);

} // namespace

namespace pycuampcor {

/// the cross-correlator of the amplitudes of count windows in the time domain on the CPU
/// (cuCorrTimeDomain)
template <>
class Correlator<Algorithm::Time, cpu::real_type, Device::CPU>
{
public:
    using real_type = cpu::real_type;   ///< the real type

    /// constructor, for nImages secondary windows of imageNX x imageNY (the sizes are not needed:
    /// the same constructor as the other correlators)
    Correlator(int imageNX, int imageNY, int nImages, cpu::stream_t stream_);
    /// compute the correlation surfaces (results) of the reference windows (templates) with the
    /// secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);

private:
    cpu::stream_t stream;   ///< the stream (not used)
};

} // namespace
//end of file
