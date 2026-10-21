/**
 * @file  cuCorrCoherence.h
 * @brief The coherent cross-correlator of the complex windows on the CUDA device:
 *     Correlator<Algorithm::Coherence, complex_type, Device::CUDA>
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"
#include "cuArrays.h"
#include "cufftPlan.h"

namespace pycuampcor {

/// the coherent cross-correlator of count complex windows on the CUDA device (cufft): the squared
/// coherence of the reference windows with the secondary windows at each shift
template <>
class Correlator<Algorithm::Coherence, cuda::complex_type, Device::CUDA>
{
public:
    using real_type = cuda::real_type;         ///< the real type
    using complex_type = cuda::complex_type;   ///< the complex type

    /// constructor, for nImages secondary windows of imageNX x imageNY
    Correlator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    /// destructor
    ~Correlator();
    /// compute the squared coherence surfaces (results) of the reference windows (templates) with
    /// the secondary windows (images): |sum conj(t - mean t) s|^2 / (sum |t - mean t|^2
    /// (sum |s|^2 - |sum s|^2 / n)), the sums over the n pixels of the template at each shift
    /// @note the squared coherence is band limited (as the complex correlation), so that its
    ///     surfaces can be oversampled (with FFTs) around the peaks; for speckle, it is comparable
    ///     with the correlation of the amplitudes
    void execute(ArraysView<const complex_type> templates, ArraysView<const complex_type> images,
        ArraysView<real_type> results);

private:
    cuda::CufftPlan plan;                    ///< the handle of the (forward and inverse) c2c fft
    cuda::cuArrays<complex_type> workT;      ///< the padded reference windows, their spectra, then the correlations
    cuda::cuArrays<complex_type> workS;      ///< the spectra of the secondary windows
    cuda::cuArrays<real_type> referenceSum2; ///< the sums of the squared magnitudes of the reference windows
    cuda::cuArrays<real_type> sat;           ///< the sum area tables of the real parts of the secondary windows
    cuda::cuArrays<real_type> satImag;       ///< the same, of the imaginary parts
    cuda::cuArrays<real_type> sat2;          ///< the same, of the squared magnitudes
    cudaStream_t stream;                     ///< the cuda stream
};

} // namespace
// end of file
