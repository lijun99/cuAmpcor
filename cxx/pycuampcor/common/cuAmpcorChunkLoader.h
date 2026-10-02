/*
 * @file  cuAmpcorChunkLoader.h
 * @brief Load chunks of reference and secondary images, shared by one or more layers
 *
 * A layer is a set of parameters (e.g., template/search window sizes) applied to the same
 * images and the same chunk partition (numberWindow*, numberWindow*InChunk). Each chunk
 * is loaded once, covering the windows of all layers, and each layer's processor then
 * copies its windows from the loaded chunk.
 */

#ifndef __CUAMPCORCHUNKLOADER_H
#define __CUAMPCORCHUNKLOADER_H

#include "backend.h"
#include "SlcImage.h"
#include "data_types.h"
#include "cuArrays.h"
#include "cuAmpcorParameter.h"
#include <memory>
#include <vector>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/// the area of each chunk in an image to be loaded, covering the windows of all layers
struct cuAmpcorChunkFootprint {
    std::vector<int> startDown, startAcross, height, width; ///< for each chunk
    int maxHeight = 0, maxWidth = 0;                          ///< the largest chunk size
};

/// a chunk of an image loaded to device (or work) memory
struct cuAmpcorLoadedChunk {
    cuArrays<image_complex_type> *complexData = nullptr; ///< for complex images
    cuArrays<image_real_type> *realData = nullptr;       ///< for real images
    int startDown = 0, startAcross = 0;  ///< first pixel of the chunk in the image
    int height = 0, width = 0;           ///< chunk size; 0 if entirely outside the image
    bool empty() const { return height == 0 || width == 0; }
};

/// chunks of the reference and secondary images loaded to device (or work) memory
struct cuAmpcorChunk {
    cuAmpcorLoadedChunk reference, secondary;
};

/// load chunks of the reference and secondary images (one loader per worker/stream)
class cuAmpcorChunkLoader {
public:
    /// footprints of the chunks covering all layers
    /// @param layers parameters of layers, set up with the starting pixels, sharing the chunk partition
    static std::pair<cuAmpcorChunkFootprint, cuAmpcorChunkFootprint>
        footprints(const std::vector<const cuAmpcorParameter *> &layers);

    cuAmpcorChunkLoader(const cuAmpcorChunkFootprint &reference, int referenceDataType, SlcImage *referenceImage,
        const cuAmpcorChunkFootprint &secondary, int secondaryDataType, SlcImage *secondaryImage,
        stream_t stream);

    /// load the idxChunk-th chunk (asynchronously on the stream)
    const cuAmpcorChunk &load(int idxChunk);

private:
    // an image and the buffer to load its chunks
    struct Source {
        const cuAmpcorChunkFootprint &footprint;
        SlcImage *image;
        std::unique_ptr<cuArrays<image_complex_type>> complexBuffer;
        std::unique_ptr<cuArrays<image_real_type>> realBuffer;
        // host memory to stage the chunks, copied asynchronously to the device (nullptr for CPU),
        // and the event marking the end of its last copy
        void *staging = nullptr;
        event_t stagingDone;
        Source(const cuAmpcorChunkFootprint &, int dataType, SlcImage *);
        ~Source();
        Source(const Source &) = delete;
        Source &operator=(const Source &) = delete;
        void load(int idxChunk, cuAmpcorLoadedChunk &chunk, stream_t stream);
    };
    Source reference, secondary;
    stream_t stream;
    cuAmpcorChunk chunk;
};

} // namespace

#endif //__CUAMPCORCHUNKLOADER_H
