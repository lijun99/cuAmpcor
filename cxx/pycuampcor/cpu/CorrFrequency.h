/**
 * @file  CorrFrequency.h
 * @brief A class performs cross correlation in frequency domain (CPU backend)
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "Arrays.h"
#include "fftwUtil.h"

// dependencies

namespace pycuampcor::cpu {

/// the cross correlator of a batch of windows in the frequency domain (fftw)
class cuFreqCorrelator
{
private:
    FftwPlan forwardPlan;               ///< the handle of the forward (r2c) fft
    FftwPlan backwardPlan;              ///< the handle of the backward (c2r) fft
    array_type<complex_type> workFM;    ///< the spectra of the reference windows, then the products
    array_type<complex_type> workFS;    ///< the spectra of the secondary windows
    array_type<real_type> workT;        ///< the padded reference windows, then the correlation surfaces
    stream_t stream;                    ///< the stream (not used)

public:
    /// constructor, for nImages windows of imageNX x imageNY (the size of the secondary windows)
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, stream_t stream_);
    /// destructor
    ~cuFreqCorrelator();
    /// compute the correlation surfaces (results) of the reference windows (templates, zero padded
    /// to the size of the secondary windows) with the secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);
};

} // namespace
// end of file
