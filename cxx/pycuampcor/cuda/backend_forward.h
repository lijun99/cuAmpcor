/**
 * @file  backend_forward.h
 * @brief Forward declarations of the CUDA backend (see forward.h)
 */

#pragma once

// the backend hooks of the shared code: stream_t, event_t, ...
#include "backend.h"

namespace pycuampcor::cuda {
    // batches of 2D arrays, in the device memory (cuArrays.h)
    template <typename T>
    class cuArrays;
    // a cufft plan (cufftPlan.h)
    class CufftPlan;
    // fft oversamplers (cuOverSampler.h) and sinc oversampler (cuSincOverSampler.h)
    class cuOverSamplerC2C;
    class cuOverSamplerR2R;
    class cuSincOverSamplerR2R;
    // cross-correlation in the frequency domain (cuCorrFrequency.h)
    class cuFreqCorrelator;
    // normalizers of the correlation surfaces (cuCorrNormalizer.h)
    class cuNormalizeProcessor;
    template <int Size>
    class cuNormalizeFixed;
    class cuNormalizeSAT;
}

//end of file
