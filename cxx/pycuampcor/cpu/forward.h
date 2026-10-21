/**
 * @file  forward.h
 * @brief Forward declarations of the CPU package, in namespace pycuampcor::cpu
 */

#pragma once

#include "externals.h"
// the shared code: parameters, images, ...
#include "../common/forward.h"

namespace pycuampcor::cpu {
    // batches of 2D arrays, in the work (host) memory (cuArrays.h)
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
    class cuNormalizeSAT;
    // the chunks of the images loaded for processing (cuAmpcorChunkLoader.h)
    struct cuAmpcorLoadedChunk;
    struct cuAmpcorChunk;
    class cuAmpcorChunkLoader;
    // the controller, running the processors over the chunks (cuAmpcorController.h)
    class cuAmpcorController;
}

//end of file
