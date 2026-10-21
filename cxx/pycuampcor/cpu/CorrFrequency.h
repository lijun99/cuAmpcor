/**
 * @file  CorrFrequency.h
 * @brief The cross-correlator of the windows in the frequency domain on the CPU:
 *     Correlator<Algorithm::Frequency, real_type, Device::CPU>
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Correlator.h"
#include "Arrays.h"
#include "fftwUtil.h"

namespace pycuampcor {

/// the cross-correlator of the amplitudes of count windows in the frequency domain on the
/// CPU (fftw)
template <>
class Correlator<Algorithm::Frequency, cpu::real_type, Device::CPU>
{
public:
    using real_type = cpu::real_type;         ///< the real type
    using complex_type = cpu::complex_type;   ///< the complex type (of the spectra)

    /// constructor, for nImages secondary windows of imageNX x imageNY
    Correlator(int imageNX, int imageNY, int nImages, cpu::stream_t stream_);
    /// destructor
    ~Correlator();
    /// compute the correlation surfaces (results) of the reference windows (templates, zero padded
    /// to the size of the secondary windows) with the secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);

private:
    cpu::FftwPlan forwardPlan;          ///< the handle of the forward (r2c) fft
    cpu::FftwPlan backwardPlan;         ///< the handle of the backward (c2r) fft
    cpu::Arrays<complex_type> workFM;   ///< the spectra of the reference windows, then the products
    cpu::Arrays<complex_type> workFS;   ///< the spectra of the secondary windows
    cpu::Arrays<real_type> workT;       ///< the padded reference windows, then the correlation surfaces
    cpu::stream_t stream;               ///< the stream (not used)
};

} // namespace
// end of file
