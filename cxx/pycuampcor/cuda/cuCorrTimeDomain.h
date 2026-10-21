/**
 * @file  cuCorrTimeDomain.h
 * @brief Cross-correlation in the time (spatial) domain (CUDA backend)
 *
 * The kernel launcher cuCorrTimeDomain, and the correlator Correlator<Algorithm::Time, real_type,
 * Device::CUDA> on it. Implemented in cuCorrTimeDomain.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"

namespace pycuampcor::cuda {

/// cross-correlate the templates (reference windows) with the images (secondary windows) in the
/// time domain: the k-th surface of results, of (image - template + 1) pixels each way, from the k-th
/// template and image
void cuCorrTimeDomain(ArraysView<const real_type> templates, ArraysView<const real_type> images,
    ArraysView<real_type> results, stream_t stream);

} // namespace

namespace pycuampcor {

/// the cross-correlator of the amplitudes of count windows in the time domain on the CUDA device
/// (cuCorrTimeDomain)
template <>
class Correlator<Algorithm::Time, cuda::real_type, Device::CUDA>
{
public:
    using real_type = cuda::real_type;   ///< the real type

    /// constructor, for nImages secondary windows of imageNX x imageNY (the sizes are not needed:
    /// the same constructor as the other correlators)
    Correlator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    /// compute the correlation surfaces (results) of the reference windows (templates) with the
    /// secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);

private:
    cudaStream_t stream;   ///< the cuda stream
};

} // namespace
//end of file
