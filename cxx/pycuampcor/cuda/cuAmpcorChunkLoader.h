/**
 * @file  cuAmpcorChunkLoader.h
 * @brief Load the chunks of the reference and secondary images to the device memory (CUDA)
 *
 * Each chunk is loaded once for all layers (see cuAmpcorLayers.h), covering their windows;
 * each layer's processor then copies its windows from the loaded chunk. A chunk is read to
 * page-locked host memory (staging) and copied asynchronously to the device, so that the
 * stream can go on queuing work (and the other streams can load) during the copy.
 */

#pragma once

#include "forward.h"
#include "cuAmpcorLayers.h"

namespace pycuampcor::cuda {

/// a chunk of an image loaded to the device memory
struct cuAmpcorLoadedChunk {
    cuArrays<image_complex_type> *complexData = nullptr; ///< for complex images
    cuArrays<image_real_type> *realData = nullptr;       ///< for real images
    int startDown = 0, startAcross = 0;  ///< first pixel of the chunk in the image
    int height = 0, width = 0;           ///< chunk size; 0 if entirely outside the image

    /// whether the chunk is entirely outside the image (nothing is loaded)
    bool empty() const { return height == 0 || width == 0; }
};

/// the chunks of the reference and secondary images loaded to the device memory
struct cuAmpcorChunk {
    cuAmpcorLoadedChunk reference;  ///< the reference chunk
    cuAmpcorLoadedChunk secondary;  ///< the secondary chunk
};

/// load the chunks of the reference and secondary images (one loader per stream)
class cuAmpcorChunkLoader {
public:
    /// a loader with the buffers for the largest chunks
    /// @param reference the footprints of the reference chunks (see chunkFootprints)
    /// @param referenceDataType the pixel type of the reference image
    /// @param referenceImage the reference image
    /// @param secondary, secondaryDataType, secondaryImage the same for the secondary image
    /// @param stream the stream on which the chunks are copied to the device
    cuAmpcorChunkLoader(const cuAmpcorChunkFootprint &reference, DataType referenceDataType,
        SlcImage *referenceImage, const cuAmpcorChunkFootprint &secondary,
        DataType secondaryDataType, SlcImage *secondaryImage, cudaStream_t stream);

    /// load the idxChunk-th chunk of both images (the copies to the device are queued on the stream)
    /// @return the loaded chunks, valid until the next load
    const cuAmpcorChunk &load(int idxChunk);

private:
    /// an image, the device buffer to load its chunks, and the staging memory
    struct Source {
        const cuAmpcorChunkFootprint &footprint;   ///< the chunks to load
        SlcImage *image;                           ///< the image
        std::unique_ptr<cuArrays<image_complex_type>> complexBuffer;  ///< for complex images
        std::unique_ptr<cuArrays<image_real_type>> realBuffer;        ///< for real images
        /// page-locked host memory for the largest chunk, copied asynchronously to the device;
        /// nullptr if it can't be allocated (then the chunks are copied from pageable memory)
        void *staging = nullptr;
        /// the event marking the end of the last copy from the staging memory
        cudaEvent_t stagingDone = nullptr;

        /// a source with the device buffer and the staging memory for the largest chunk
        Source(const cuAmpcorChunkFootprint &footprint, DataType dataType, SlcImage *image);
        /// wait for the last copy, and free the staging memory
        ~Source();
        Source(const Source &) = delete;
        Source &operator=(const Source &) = delete;
        /// load the idxChunk-th chunk into the device buffer (on the stream), and describe it in chunk
        void load(int idxChunk, cuAmpcorLoadedChunk &chunk, cudaStream_t stream);
    };
    Source reference;       ///< the reference image
    Source secondary;       ///< the secondary image
    cudaStream_t stream;    ///< the stream of the copies
    cuAmpcorChunk chunk;    ///< the last loaded chunks
};

} // namespace
//end of file
