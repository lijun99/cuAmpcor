/*
 * @file  CorrFrequency.h
 * @brief A class performs cross correlation in frequency domain (CPU backend)
 */

#pragma once

#include "forward.h"
#include "fftwUtil.h"

// dependencies

namespace pycuampcor::cpu {

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    FftwPlan forwardPlan;
    FftwPlan backwardPlan;
    // work data
    std::unique_ptr<cuArrays<complex_type>> workFM;
    std::unique_ptr<cuArrays<complex_type>> workFS;
    std::unique_ptr<cuArrays<real_type>> workT;
    // stream (not used)
    stream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, stream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(cuArrays<real_type> *templates, cuArrays<real_type> *images, cuArrays<real_type> *results);
};

} // namespace
// end of file
