/**
 * @file  arrayTraits.h
 * @brief The types and workers that go with the arrays of the CPU package (see cuAmpcorArrayTraits.h)
 */

#pragma once

#include "forward.h"
#include "cuAmpcorArrayTraits.h"
#include "cuCorrNormalizer.h"

namespace pycuampcor {

/// the types and workers that go with the arrays of the CPU package, for the processors
template <>
struct array_traits<cpu::cuArrays> {
    // the element types
    using real_type = cpu::real_type;
    using complex_type = cpu::complex_type;
    using real2_type = cpu::real2_type;
    using real3_type = cpu::real3_type;
    using int2_type = cpu::int2;
    // the stream on which the kernels are queued
    using stream_type = cpu::stream_t;
    // the chunks of the images loaded by the chunk loader
    using loaded_chunk_type = cpu::cuAmpcorLoadedChunk;
    using chunk_type = cpu::cuAmpcorChunk;
    // the workers
    using freq_correlator_type = cpu::cuFreqCorrelator;
    using oversampler_c2c_type = cpu::cuOverSamplerC2C;
    using oversampler_r2r_type = cpu::cuOverSamplerR2R;
    using sinc_oversampler_type = cpu::cuSincOverSamplerR2R;
    using normalizer_type = cpu::cuNormalizeProcessor;

    /// an int2_type (x: down, y: across)
    static int2_type make_int2(int x, int y) { return cpu::make_int2(x, y); }

    /// a normalizer for the correlation surfaces of count windows of height x width
    static std::unique_ptr<normalizer_type> newNormalizer(int height, int width, int count)
    {
        return cpu::newCuNormalizer(height, width, count);
    }
};

} // namespace
//end of file
