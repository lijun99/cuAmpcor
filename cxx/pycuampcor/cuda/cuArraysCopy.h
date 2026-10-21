/**
 * @file  cuArraysCopy.h
 * @brief Copy, extract, insert and convert batches of images (CUDA backend)
 *
 * Implemented in cuArraysCopy.cu.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// copy the windows from a chunk of a complex image to a batch: the window k starts at the pixel
/// (offsetH[k], offsetW[k]) of the chunk, and its pixels outside the chunk are zeros
void cuArraysCopyToBatchWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t stream);

/// the same, with the amplitudes of the pixels (as complex values)
void cuArraysCopyToBatchAbsWithOffset(ArraysView<const image_complex_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t stream);

/// the same, from a chunk of a real image (as complex values)
void cuArraysCopyToBatchWithOffsetR2C(ArraysView<const image_real_type> chunk,
    ArraysView<complex_type> windows, ArraysView<const int> offsetH, ArraysView<const int> offsetW,
    stream_t stream);

/// extract the k-th array of imagesOut from the k-th array of imagesIn, starting at its pixel offsets[k]
void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> offsets, stream_t stream);
/// the same, for complex arrays
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    ArraysView<const int2> offsets, stream_t stream);

/// extract the k-th array of imagesOut from the k-th array of imagesIn, starting at its pixel offset
void cuArraysCopyExtract(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t stream);
/// the same, the real parts of complex arrays
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t stream);
/// the same, for complex arrays
void cuArraysCopyExtract(ArraysView<const complex_type> imagesIn, ArraysView<complex_type> imagesOut,
    int2 offset, stream_t stream);
/// the same, for arrays of real3_type
void cuArraysCopyExtract(ArraysView<const real3_type> imagesIn, ArraysView<real3_type> imagesOut,
    int2 offset, stream_t stream);

/// the same, the amplitudes of complex arrays
void cuArraysCopyExtractAbs(ArraysView<const complex_type> imagesIn, ArraysView<real_type> imagesOut,
    int2 offset, stream_t stream);

/// insert the (first) array imageIn into the (first) array imageOut, at its pixel (offsetX, offsetY)
void cuArraysCopyInsert(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    int offsetX, int offsetY, stream_t stream);
/// the same, for arrays of real3_type
void cuArraysCopyInsert(ArraysView<const real3_type> imageIn, ArraysView<real3_type> imageOut,
    int offsetX, int offsetY, stream_t stream);
/// the same, for real arrays
void cuArraysCopyInsert(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    int offsetX, int offsetY, stream_t stream);
/// the same, for integer arrays
void cuArraysCopyInsert(ArraysView<const int> imageIn, ArraysView<int> imageOut,
    int offsetX, int offsetY, stream_t stream);

/// copy the arrays imageIn to the (larger) arrays imageOut, padded with zeros
void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<real_type> imageOut,
    stream_t stream);
/// the same, from real to complex arrays
void cuArraysCopyPadded(ArraysView<const real_type> imageIn, ArraysView<complex_type> imageOut,
    stream_t stream);
/// the same, for complex arrays
void cuArraysCopyPadded(ArraysView<const complex_type> imageIn, ArraysView<complex_type> imageOut,
    stream_t stream);

/// set the elements of the arrays to value
void cuArraysSetConstant(ArraysView<real_type> imageIn, real_type value, stream_t stream);

/// real arrays as complex ones
void cuArraysR2C(ArraysView<const real_type> image1, ArraysView<complex_type> image2,
    stream_t stream);

/// the real parts of complex arrays
void cuArraysC2R(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    stream_t stream);

/// the amplitudes of complex arrays
void cuArraysAbs(ArraysView<const complex_type> image1, ArraysView<real_type> image2,
    stream_t stream);

/// extract the arrays of imagesOut centered at the pixels maxloc of the arrays imagesIn (within them)
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<const int2> maxloc, stream_t stream);

/// the same, marking the pixels within imagesIn (1) or outside (0, with the value 0) in imagesValid
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, stream_t stream);

/// the same, the pixels at a stride (e.g., at the raw pixel spacing of an oversampled surface),
/// valid within the region of validRange pixels starting at validStart
void cuArraysCopyExtractCorr(ArraysView<const real_type> imagesIn, ArraysView<real_type> imagesOut,
    ArraysView<int> imagesValid, ArraysView<const int2> maxloc, int stride, int2 validStart,
    int2 validRange, stream_t stream);

} // namespace
//end of file
