/**
 * @file  cuAmpcorChunkLoader.cpp
 * @brief Load the chunks of the reference and secondary images to the work memory (CPU)
 */

#include "cuAmpcorChunkLoader.h"
#include "cuArrays.h"
#include "cuAmpcorOptions.h"
#include "SlcImage.h"

namespace pycuampcor::cpu {

/**
 * A source with a buffer for the largest chunk of the footprint
 * @param footprint_ the chunks to load
 * @param dataType the pixel type of the image: complex or real
 * @param image_ the image
 */
cuAmpcorChunkLoader::Source::Source(const cuAmpcorChunkFootprint &footprint_, DataType dataType,
    SlcImage *image_)
    : footprint(footprint_), image(image_)
{
    // allocate the buffer once, for the largest chunk
    const int height = std::max(footprint.maxHeight, 1);
    const int width = std::max(footprint.maxWidth, 1);
    if (dataType == DataType::Complex64) {
        complexBuffer = std::make_unique<cuArrays<image_complex_type>>(height, width);
        complexBuffer->allocate();
    }
    else {
        realBuffer = std::make_unique<cuArrays<image_real_type>>(height, width);
        realBuffer->allocate();
    }
}

/**
 * Load the idxChunk-th chunk into the buffer
 * @param idxChunk the index of the chunk
 * @param[out] chunk the description of the loaded chunk: its buffer, position and size
 */
void cuAmpcorChunkLoader::Source::load(int idxChunk, cuAmpcorLoadedChunk &chunk)
{
    chunk.complexData = complexBuffer.get();
    chunk.realData = realBuffer.get();
    chunk.startDown = footprint.startDown[idxChunk];
    chunk.startAcross = footprint.startAcross[idxChunk];
    chunk.height = footprint.height[idxChunk];
    chunk.width = footprint.width[idxChunk];
    // nothing to load for chunks outside the image (their windows are set to zero)
    if (chunk.empty()) return;
    // the work memory is host memory: read the chunk directly into the buffer
    void *buffer = complexBuffer ? static_cast<void *>(complexBuffer->devData)
                                 : static_cast<void *>(realBuffer->devData);
    image->loadToHost(buffer, chunk.startDown, chunk.startAcross, chunk.height, chunk.width);
}

/**
 * A loader with the buffers for the largest chunks of the reference and secondary images
 */
cuAmpcorChunkLoader::cuAmpcorChunkLoader(
    const cuAmpcorChunkFootprint &referenceFootprint, DataType referenceDataType,
    SlcImage *referenceImage, const cuAmpcorChunkFootprint &secondaryFootprint,
    DataType secondaryDataType, SlcImage *secondaryImage)
    : reference(referenceFootprint, referenceDataType, referenceImage),
      secondary(secondaryFootprint, secondaryDataType, secondaryImage)
{
}

/**
 * Load the idxChunk-th chunk of both images
 * @return the loaded chunks, valid until the next load
 */
const cuAmpcorChunk &cuAmpcorChunkLoader::load(int idxChunk)
{
    reference.load(idxChunk, chunk.reference);
    secondary.load(idxChunk, chunk.secondary);
    return chunk;
}

} // namespace
//end of file
