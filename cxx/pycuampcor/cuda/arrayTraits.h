/**
 * @file  arrayTraits.h
 * @brief The types and workers that go with the arrays of the CUDA package (see cuAmpcorArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"

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
    // the interfaces of the normalizer and of the surface oversampler (common/), implemented by the package
    using normalizer_type = cuNormalizer<cuda::cuArrays>;
    using surface_oversampler_type = cuSurfaceOverSampler<cuda::cuArrays>;

    /// an int2_type (x: down, y: across)
    static int2_type make_int2(int x, int y) { return ::make_int2(x, y); }

    /// a normalizer for the correlation surfaces of count windows of height x width
    /// (defined with the processors, processors.cpp)
    static std::unique_ptr<normalizer_type> newNormalizer(int height, int width, int count);

    /// an oversampler (FFT or sinc, by method) of count correlation surfaces of height x width
    /// around their peaks, by factor; with rawFactor, the oversampling of the windows (the sinc
    /// oversampler shifts the peaks by factor*rawFactor)
    static std::unique_ptr<surface_oversampler_type> newSurfaceOverSampler(OversamplingMethod method,
        int height, int width, int factor, int rawFactor, int count, stream_type stream);
};

} // namespace
//end of file
