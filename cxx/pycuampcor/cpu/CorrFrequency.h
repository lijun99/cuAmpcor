/*
 * @file  CorrFrequency.h
 * @brief A class performs cross correlation in frequency domain (CPU backend)
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"
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
    std::unique_ptr<array_type<complex_type>> workFM;
    std::unique_ptr<array_type<complex_type>> workFS;
    std::unique_ptr<array_type<real_type>> workT;
    // stream (not used)
    stream_t stream;

public:
    // constructor
    cuFreqCorrelator(int imageNX, int imageNY, int nImages, stream_t stream_);
    // destructor
    ~cuFreqCorrelator();
    // executor
    void execute(ArraysView<const real_type> templates, ArraysView<const real_type> images,
        ArraysView<real_type> results);
};

} // namespace
// end of file
