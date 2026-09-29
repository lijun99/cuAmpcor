/*
 * @file  cuCorrFrequency.h
 * @brief A class performs cross correlation in frequency domain (CPU backend)
 */

// code guard
#ifndef __CUCORRFREQUENCY_H
#define __CUCORRFREQUENCY_H

// dependencies
#include "cuArrays.h"
#include "data_types.h"
#include "fftwUtil.h"

namespace pycuampcor::cpu {

class cuFreqCorrelator
{
private:
    // handles for forward/backward fft
    fftw_plan_type forwardPlan;
    fftw_plan_type backwardPlan;
    // work data
    cuArrays<complex_type> *workFM;
    cuArrays<complex_type> *workFS;
    cuArrays<real_type> *workT;
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

#endif //__CUCORRFREQUENCY_H
// end of file
