/**
 * @file  traits.h
 * @brief The types and workers that go with the arrays of the CUDA package (see ArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"

namespace pycuampcor {

/// the types and workers that go with the arrays of the CUDA package, for the processors
template <>
struct array_traits<cuda::cuArrays> {
    /// the device of the arrays: the CUDA device
    static constexpr Device device = Device::CUDA;
    // the element types
    using real_type = cuda::real_type;         ///< the real type
    using complex_type = cuda::complex_type;   ///< the complex type
    using real2_type = cuda::real2_type;       ///< a pair of reals (e.g., the offsets)
    using real3_type = cuda::real3_type;       ///< a triple of reals (e.g., the covariances)
    using int2_type = ::int2;                  ///< a pair of ints (e.g., the window positions)
    /// the stream on which the kernels are queued
    using stream_type = cuda::stream_t;
    // the chunks of the images loaded by the chunk loader
    /// a chunk of an image loaded to the device memory
    using loaded_chunk_type = cuda::LoadedChunk;
    /// the chunks of the reference and secondary images loaded to the device memory
    using chunk_type = cuda::Chunk;
    // the workers
    /// the cross-correlation of the windows in the frequency domain
    using freq_correlator_type = cuda::cuFreqCorrelator;
    // the interfaces of the normalizer and of the surface oversampler (common/), implemented by the package
    /// the normalizer of the correlation surfaces
    using normalizer_type = Normalizer<cuda::cuArrays>;
    /// the oversampler of the correlation surfaces around their peaks
    using surface_oversampler_type = SurfaceOverSampler<cuda::cuArrays>;

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
