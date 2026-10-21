/**
 * @file  arrayTraits.h
 * @brief The types and workers that go with the arrays of the CUDA package (see cuAmpcorArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"
#include "cuCorrNormalizer.h"

namespace pycuampcor {

/// the types and workers that go with the arrays of the CUDA package, for the processors
template <>
struct array_traits<cuda::cuArrays> {
    // the element types
    using real_type = cuda::real_type;
    using complex_type = cuda::complex_type;
    using real2_type = cuda::real2_type;
    using real3_type = cuda::real3_type;
    using int2_type = ::int2;
    // the stream on which the kernels are queued
    using stream_type = cuda::stream_t;
    // the chunks of the images loaded by the chunk loader
    using loaded_chunk_type = cuda::cuAmpcorLoadedChunk;
    using chunk_type = cuda::cuAmpcorChunk;
    // the workers
    using freq_correlator_type = cuda::cuFreqCorrelator;
    using oversampler_c2c_type = cuda::cuOverSamplerC2C;
    using oversampler_r2r_type = cuda::cuOverSamplerR2R;
    using sinc_oversampler_type = cuda::cuSincOverSamplerR2R;
    using normalizer_type = cuda::cuNormalizeProcessor;

    /// an int2_type (x: down, y: across)
    static int2_type make_int2(int x, int y) { return ::make_int2(x, y); }

    /// a normalizer for the correlation surfaces of count windows of height x width
    static std::unique_ptr<normalizer_type> newNormalizer(int height, int width, int count)
    {
        return cuda::newCuNormalizer(height, width, count);
    }
};

} // namespace
//end of file
