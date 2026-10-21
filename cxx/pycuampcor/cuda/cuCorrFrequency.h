/**
 * @file  cuCorrFrequency.h
 * @brief The cross-correlator of the windows in the frequency domain on the CUDA device:
 *     Correlator<Algorithm::Frequency, real_type, Device::CUDA>
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"
#include "cuArrays.h"
#include "cufftPlan.h"

namespace pycuampcor {

/// the cross-correlator of the amplitudes of count windows in the frequency domain on the
/// CUDA device (cufft)
template <>
class Correlator<Algorithm::Frequency, cuda::real_type, Device::CUDA>
{
public:
    using real_type = cuda::real_type;         ///< the real type
    using complex_type = cuda::complex_type;   ///< the complex type (of the spectra)

    /// constructor, for nImages secondary windows of imageNX x imageNY
    Correlator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    /// destructor
    ~Correlator();
    /// compute the correlation surfaces (results) of the reference windows (templates, zero padded
    /// to the size of the secondary windows) with the secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);

private:
    cuda::CufftPlan forwardPlan;           ///< the handle of the forward (r2c) fft
    cuda::CufftPlan backwardPlan;          ///< the handle of the backward (c2r) fft
    cuda::cuArrays<complex_type> workFM;   ///< the spectra of the reference windows, then the products
    cuda::cuArrays<complex_type> workFS;   ///< the spectra of the secondary windows
    cuda::cuArrays<real_type> workT;       ///< the padded reference windows, then the correlation surfaces
    cudaStream_t stream;                   ///< the cuda stream
};

} // namespace
// end of file
