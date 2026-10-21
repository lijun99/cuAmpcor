/**
 * @file  forward.h
 * @brief Forward declarations of the CUDA package, in namespace pycuampcor::cuda
 */

#pragma once

#include "externals.h"
// the shared code: parameters, images, ...
#include "../common/forward.h"

namespace pycuampcor::cuda {
    // batches of 2D arrays, in the device memory (cuArrays.h)
    template <typename T>
    class cuArrays;
    /// the arrays of the package: batches of 2D arrays of T (as array_type in the cpu package)
    template <typename T>
    using array_type = cuArrays<T>;
    // a cufft plan (cufftPlan.h)
    class CufftPlan;
    // fft oversamplers (cuOverSampler.h) and sinc oversampler (cuSincOverSampler.h)
    class cuOverSamplerC2C;
    class cuOverSamplerR2R;
    class cuSincOverSamplerR2R;
    // cross-correlation in the frequency domain (cuCorrFrequency.h)
    class cuFreqCorrelator;
    // normalizers of the correlation surfaces (cuCorrNormalizer.h)
    template <int Size>
    class cuNormalizeFixed;
    class cuNormalizeSAT;
    // the chunks of the images loaded for processing (cuChunkLoader.h)
    struct LoadedChunk;
    struct Chunk;
    class ChunkLoader;
    // the controller, running the processors over the chunks (cuController.h)
    class Controller;
}

//end of file
