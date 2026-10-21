/**
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
#include "cuArrays.h"
#include "cufftPlan.h"

// dependencies

namespace pycuampcor::cuda {

/// the cross correlator of a batch of windows in the frequency domain (cufft)
class cuFreqCorrelator
{
private:
    CufftPlan forwardPlan;              ///< the handle of the forward (r2c) fft
    CufftPlan backwardPlan;             ///< the handle of the backward (c2r) fft
    cuArrays<complex_type> workFM;      ///< the spectra of the reference windows, then the products
    cuArrays<complex_type> workFS;      ///< the spectra of the secondary windows
    cuArrays<real_type> workT;          ///< the padded reference windows, then the correlation surfaces
    cudaStream_t stream;                ///< the cuda stream

public:
    /// constructor, for nImages windows of imageNX x imageNY (the size of the secondary windows)
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    /// destructor
    ~cuFreqCorrelator();
    /// compute the correlation surfaces (results) of the reference windows (templates, zero padded
    /// to the size of the secondary windows) with the secondary windows (images)
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);
};

} // namespace
// end of file
