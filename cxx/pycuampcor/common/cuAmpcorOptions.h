/**
 * @file  cuAmpcorOptions.h
 * @brief Options of the ampcor parameters
 *
 * The values are those of the integer enums in pycuampcor.options (python).
 */

#pragma once

#include <cstddef>

namespace pycuampcor {

/// Cross-correlation domain
enum class Algorithm : int {
    Frequency = 0,  ///< in the frequency domain, with FFTs
    Time = 1,       ///< in the time (spatial) domain
};

/// Cross-correlation workflow
enum class Workflow : int {
    TwoPass = 0,  ///< a first pass without the antialiasing oversampling
    OnePass = 1,  ///< one pass with the antialiasing oversampling
};

/// Removal of the phase ramp before the antialiasing oversampling
enum class DerampMethod : int {
    Magnitude = 0,  ///< take the amplitudes (e.g., for TOPS)
    Complex = 1,    ///< remove a linear phase ramp, estimated from the average phase difference
    None = 2,       ///< no deramping
};

/// Axis (or axes) of the phase ramp to remove
enum class DerampAxis : int {
    Down = 0,    ///< down (azimuth)
    Across = 1,  ///< across (range)
    Both = 2,    ///< both
};

/// Oversampling method of the correlation surface
enum class OversamplingMethod : int {
    FFT = 0,   ///< FFT
    Sinc = 1,  ///< sinc interpolation
};

/// Pixel type of an image (the value is the number of floats in a pixel)
enum class DataType : int {
    Float32 = 1,    ///< real (amplitude) images
    Complex64 = 2,  ///< complex images
};

/// size of a pixel in bytes
inline std::size_t pixelSize(DataType type)
{
    return static_cast<std::size_t>(type) * sizeof(float);
}

} // namespace
//end of file
