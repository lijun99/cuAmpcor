/**
 * @file  cuAmpcorChunkLoader.h
 * @brief Load the chunks of the reference and secondary images to the work memory (CPU)
 *
 * Each chunk is loaded once for all layers (see cuAmpcorLayers.h), covering their windows;
 * each layer's processor then copies its windows from the loaded chunk. On the CPU, the work
 * memory is host memory: the chunks are read directly into it.
 */

#pragma once

#include "forward.h"
#include "cuAmpcorLayers.h"

namespace pycuampcor::cpu {

/// a chunk of an image loaded to the work memory
struct cuAmpcorLoadedChunk {
    cuArrays<image_complex_type> *complexData = nullptr; ///< for complex images
    cuArrays<image_real_type> *realData = nullptr;       ///< for real images
    int startDown = 0, startAcross = 0;  ///< first pixel of the chunk in the image
    int height = 0, width = 0;           ///< chunk size; 0 if entirely outside the image

    /// whether the chunk is entirely outside the image (nothing is loaded)
    bool empty() const { return height == 0 || width == 0; }
};

/// the chunks of the reference and secondary images loaded to the work memory
struct cuAmpcorChunk {
    cuAmpcorLoadedChunk reference;  ///< the reference chunk
    cuAmpcorLoadedChunk secondary;  ///< the secondary chunk
};

/// load the chunks of the reference and secondary images (one loader per thread)
class cuAmpcorChunkLoader {
public:
    /// a loader with the buffers for the largest chunks
    /// @param reference the footprints of the reference chunks (see chunkFootprints)
    /// @param referenceDataType the pixel type of the reference image
    /// @param referenceImage the reference image
    /// @param secondary, secondaryDataType, secondaryImage the same for the secondary image
    cuAmpcorChunkLoader(const cuAmpcorChunkFootprint &reference, DataType referenceDataType,
        SlcImage *referenceImage, const cuAmpcorChunkFootprint &secondary,
        DataType secondaryDataType, SlcImage *secondaryImage);

    /// load the idxChunk-th chunk of both images
    /// @return the loaded chunks, valid until the next load
    const cuAmpcorChunk &load(int idxChunk);

private:
    /// an image and the buffer to load its chunks
    struct Source {
        const cuAmpcorChunkFootprint &footprint;   ///< the chunks to load
        SlcImage *image;                           ///< the image
        std::unique_ptr<cuArrays<image_complex_type>> complexBuffer;  ///< for complex images
        std::unique_ptr<cuArrays<image_real_type>> realBuffer;        ///< for real images

        /// a source with a buffer for the largest chunk of the footprint
        Source(const cuAmpcorChunkFootprint &footprint, DataType dataType, SlcImage *image);
        Source(const Source &) = delete;
        Source &operator=(const Source &) = delete;
        /// load the idxChunk-th chunk into the buffer, and describe it in chunk
        void load(int idxChunk, cuAmpcorLoadedChunk &chunk);
    };
    Source reference;      ///< the reference image
    Source secondary;      ///< the secondary image
    cuAmpcorChunk chunk;   ///< the last loaded chunks
};

} // namespace
//end of file
