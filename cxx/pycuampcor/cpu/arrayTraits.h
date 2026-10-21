/**
 * @file  arrayTraits.h
 * @brief The types and workers that go with the arrays of the CPU package (see cuAmpcorArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"

namespace pycuampcor {

/// the types and workers that go with the arrays of the CPU package, for the processors
template <>
struct array_traits<cpu::Arrays> {
    // the element types
    using real_type = cpu::real_type;         ///< the real type
    using complex_type = cpu::complex_type;   ///< the complex type
    using real2_type = cpu::real2_type;       ///< a pair of reals (e.g., the offsets)
    using real3_type = cpu::real3_type;       ///< a triple of reals (e.g., the covariances)
    using int2_type = cpu::int2;              ///< a pair of ints (e.g., the window positions)
    /// the stream on which the kernels are queued
    using stream_type = cpu::stream_t;
    // the chunks of the images loaded by the chunk loader
    /// a chunk of an image loaded to the work memory
    using loaded_chunk_type = cpu::cuAmpcorLoadedChunk;
    /// the chunks of the reference and secondary images loaded to the work memory
    using chunk_type = cpu::cuAmpcorChunk;
    // the workers
    /// the cross-correlation of the windows in the frequency domain
    using freq_correlator_type = cpu::cuFreqCorrelator;
    /// the FFT oversampler of complex windows
    using oversampler_c2c_type = cpu::cuOverSamplerC2C;
    // the interfaces of the normalizer and of the surface oversampler (common/), implemented by the package
    /// the normalizer of the correlation surfaces
    using normalizer_type = cuNormalizer<cpu::Arrays>;
    /// the oversampler of the correlation surfaces around their peaks
    using surface_oversampler_type = cuSurfaceOverSampler<cpu::Arrays>;

    /// an int2_type (x: down, y: across)
    static int2_type make_int2(int x, int y) { return cpu::make_int2(x, y); }

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
