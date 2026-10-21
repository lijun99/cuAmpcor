/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain
 */

#pragma once

// dependencies
#include "cuArrays.h"
#include "data_types.h"
#include "cufftPlan.h"

#include <memory>

namespace pycuampcor::cuda {

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    CufftPlan forwardPlan;
    CufftPlan backwardPlan;
    // work data
    std::unique_ptr<cuArrays<complex_type>> workFM;
    std::unique_ptr<cuArrays<complex_type>> workFS;
    std::unique_ptr<cuArrays<real_type>> workT;
    // cuda stream
    cudaStream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, cudaStream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results);
};

} // namespace
// end of file
