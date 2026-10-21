/**
 * @file  backend_forward.h
 * @brief Forward declarations of the CPU backend (see forward.h)
 */

#pragma once

// the backend hooks of the shared code: stream_t, event_t, ...
#include "backend.h"

namespace pycuampcor::cpu {
    // batches of 2D arrays, in the work memory (cuArrays.h)
    template <typename T>
    class cuArrays;
    // an fftw plan (fftwUtil.h)
    class FftwPlan;
    // fft oversamplers (cuOverSampler.h) and sinc oversampler (cuSincOverSampler.h)
    class cuOverSamplerC2C;
    class cuOverSamplerR2R;
    class cuSincOverSamplerR2R;
    // cross-correlation in the frequency domain (cuCorrFrequency.h)
    class cuFreqCorrelator;
    // normalizers of the correlation surfaces (cuCorrNormalizer.h)
    class cuNormalizeProcessor;
    class cuNormalizeSAT;
}

//end of file
