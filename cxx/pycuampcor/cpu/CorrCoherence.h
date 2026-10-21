/**
 * @file  CorrCoherence.h
 * @brief The coherent cross-correlator of the complex windows on the CPU:
 *     Correlator<Algorithm::Coherence, complex_type, Device::CPU>
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"
#include "Arrays.h"
#include "fftwUtil.h"

namespace pycuampcor {

/// the coherent cross-correlator of count complex windows on the CPU (fftw): the squared coherence
/// of the reference windows with the secondary windows at each shift
template <>
class Correlator<Algorithm::Coherence, cpu::complex_type, Device::CPU>
{
public:
    using real_type = cpu::real_type;         ///< the real type
    using complex_type = cpu::complex_type;   ///< the complex type

    /// constructor, for nImages secondary windows of imageNX x imageNY
    Correlator(int imageNX, int imageNY, int nImages, cpu::stream_t stream_);
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
    cpu::FftwPlan forwardPlan;          ///< the handle of the forward fft
    cpu::FftwPlan backwardPlan;         ///< the handle of the backward fft
    cpu::Arrays<complex_type> workT;    ///< the padded reference windows, their spectra, then the correlations
    cpu::Arrays<complex_type> workS;    ///< the spectra of the secondary windows
    cpu::Arrays<double> sat;            ///< the sum area table of the real parts of a secondary window
    cpu::Arrays<double> satImag;        ///< the same, of the imaginary parts
    cpu::Arrays<double> sat2;           ///< the same, of the squared magnitudes
    cpu::stream_t stream;               ///< the stream (not used)
};

} // namespace
// end of file
