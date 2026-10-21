/*
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

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    CufftPlan forwardPlan;
    CufftPlan backwardPlan;
    // work data
    cuArrays<complex_type> workFM;
    cuArrays<complex_type> workFS;
    cuArrays<real_type> workT;
    // cuda stream
    cudaStream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);
};

} // namespace
// end of file
