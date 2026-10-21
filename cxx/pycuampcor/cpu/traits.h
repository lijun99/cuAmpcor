/**
 * @file  traits.h
 * @brief The types and workers that go with the arrays of the CPU package (see ArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "ArrayTraits.h"

namespace pycuampcor {

/// the types and workers that go with the arrays of the CPU package, for the processors
template <>
struct array_traits<cpu::Arrays> {
    /// the device of the arrays: the CPU (host memory)
    static constexpr Device device = Device::CPU;
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
    using loaded_chunk_type = cpu::LoadedChunk;
    /// the chunks of the reference and secondary images loaded to the work memory
    using chunk_type = cpu::Chunk;
    // the workers
    // the interfaces of the normalizer and of the surface oversampler (common/), implemented by the package
    /// the cross-correlation of the windows (by an algorithm chosen at run time)
    using correlator_type = WindowCorrelator<cpu::Arrays>;
    /// the normalizer of the correlation surfaces
    using normalizer_type = Normalizer<cpu::Arrays>;
    /// the oversampler of the correlation surfaces around their peaks
    using surface_oversampler_type = SurfaceOverSampler<cpu::Arrays>;

    /// an int2_type (x: down, y: across)
    static int2_type make_int2(int x, int y) { return cpu::make_int2(x, y); }

    /// a normalizer for the correlation surfaces of count windows of height x width
    /// (defined with the processors, processors.cpp)
    static std::unique_ptr<normalizer_type> newNormalizer(int height, int width, int count);
};

} // namespace
//end of file
