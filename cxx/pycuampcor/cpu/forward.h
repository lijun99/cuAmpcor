/**
 * @file  forward.h
 * @brief Forward declarations of the CPU package, in namespace pycuampcor::cpu
 */

#pragma once

#include "externals.h"
// the shared code: parameters, images, ...
#include "../common/forward.h"

namespace pycuampcor::cpu {
    // batches of 2D arrays, in the work (host) memory (Arrays.h)
    template <typename T>
    class Arrays;
    /// the arrays of the package: batches of 2D arrays of T, used by all its modules
    template <typename T>
    using array_type = Arrays<T>;
    // an fftw plan (fftwUtil.h)
    class FftwPlan;
    // cross-correlation in the frequency domain (CorrFrequency.h)
    class cuFreqCorrelator;
    // normalizers of the correlation surfaces (CorrNormalizer.h)
    class cuNormalizeSAT;
    // the chunks of the images loaded for processing (ChunkLoader.h)
    struct LoadedChunk;
    struct Chunk;
    class ChunkLoader;
    // the controller, running the processors over the chunks (Controller.h)
    class Controller;
}

//end of file
