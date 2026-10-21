/**
 * @file  ArrayTraits.h
 * @brief What goes with an array type: the types and workers the processors use with it
 *
 * The processors (Processor*) are written once, as templates on the array type: a family
 * of batches of 2D arrays, e.g., cpu::Arrays or cuda::cuArrays. array_traits gives the element
 * types, the stream and chunk types, and the workers (FFT correlator, oversamplers, normalizer)
 * that go with an array type; each package (cpu, cuda) specializes it for its arrays
 * (traits.h of the package).
 *
 * The kernels (e.g., cuArraysAbs) are found from the array type of their arguments (by argument
 * dependent lookup), so the processors call them as is.
 */

#pragma once

namespace pycuampcor {

/// the device the arrays (and the workers on them, e.g., OverSampler) are on
enum class Device : int {
    CPU,    ///< the host memory (cpu package)
    CUDA,   ///< the memory of a CUDA device (cuda package)
};

/// the types and workers that go with an array type arrayT (specialized by each package)
///
/// A specialization provides:
/// - device: the Device of the arrays, which selects the workers templated on it (OverSampler)
/// - real_type, complex_type, real2_type, real3_type, int2_type: the element types
/// - stream_type: the stream on which the kernels are queued
/// - loaded_chunk_type, chunk_type: the chunks of the images loaded by the chunk loader
/// - correlator_type: the cross-correlation of the windows (an abstract class, WindowCorrelator)
/// - normalizer_type: the normalizer of the correlation surfaces (an abstract class)
/// - surface_oversampler_type: the oversampler of the correlation surfaces (an abstract class)
/// - make_int2(x, y): an int2_type
/// - newNormalizer(height, width, count): a normalizer for the correlation of count windows
///   of height x width
template <template <typename> class arrayT>
struct array_traits;

} // namespace
//end of file
