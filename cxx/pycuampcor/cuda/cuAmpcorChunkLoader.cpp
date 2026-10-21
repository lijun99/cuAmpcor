/**
 * @file  cuAmpcorChunkLoader.cpp
 * @brief Load the chunks of the reference and secondary images to the device memory (CUDA)
 */

#include "cuAmpcorChunkLoader.h"
#include "cuArrays.h"
#include "cuAmpcorOptions.h"
#include "cudaError.h"
#include "logging.h"
#include "SlcImage.h"

namespace pycuampcor::cuda {

/**
 * A source with the device buffer and the staging memory for the largest chunk of the footprint
 * @param footprint_ the chunks to load
 * @param dataType the pixel type of the image: complex or real
 * @param image_ the image
 */
cuAmpcorChunkLoader::Source::Source(const cuAmpcorChunkFootprint &footprint_, DataType dataType,
    SlcImage *image_)
    : footprint(footprint_), image(image_)
{
    // allocate the device buffer once, for the largest chunk
    // (allocating/freeing device memory per chunk synchronizes the device and stalls the other streams)
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
    // page-locked host memory for the largest chunk, so that the copies to the device are asynchronous
    const size_t bytes = static_cast<size_t>(height) * width * image->pixelSize();
    if (cudaMallocHost(&staging, bytes) != cudaSuccess) {
        // e.g., limited page-locked memory: clear the error, and copy from pageable memory
        cudaGetLastError();
        staging = nullptr;
        logWarning("failed to allocate ", bytes, " bytes of page-locked memory;",
                   " using pageable memory to load the images");
    }
    checkCudaErrors(cudaEventCreateWithFlags(&stagingDone, cudaEventDisableTiming));
}

/**
 * Wait for the last copy from the staging memory, and free it
 */
cuAmpcorChunkLoader::Source::~Source()
{
    // the last copy from the staging memory must be done before the memory is freed
    if (stagingDone) {
        checkCudaErrors(cudaEventSynchronize(stagingDone));
        checkCudaErrors(cudaEventDestroy(stagingDone));
    }
    if (staging)
        checkCudaErrors(cudaFreeHost(staging));
}

/**
 * Load the idxChunk-th chunk into the device buffer
 * @param idxChunk the index of the chunk
 * @param[out] chunk the description of the loaded chunk: its buffer, position and size
 * @param stream the stream on which the copy to the device is queued
 */
void cuAmpcorChunkLoader::Source::load(int idxChunk, cuAmpcorLoadedChunk &chunk, cudaStream_t stream)
{
    chunk.complexData = complexBuffer.get();
    chunk.realData = realBuffer.get();
    chunk.startDown = footprint.startDown[idxChunk];
    chunk.startAcross = footprint.startAcross[idxChunk];
    chunk.height = footprint.height[idxChunk];
    chunk.width = footprint.width[idxChunk];
    // nothing to load for chunks outside the image (their windows are set to zero)
    if (chunk.empty()) return;

    void *buffer = complexBuffer ? static_cast<void *>(complexBuffer->devData)
                                 : static_cast<void *>(realBuffer->devData);
    // the chunk is contiguous in both host and device memory
    const size_t pitch = static_cast<size_t>(chunk.width) * image->pixelSize();
    if (staging) {
        // reuse the staging memory once its previous copy is done, then copy asynchronously,
        // so the stream can go on queuing its work (and the other streams can load) during the copy
        checkCudaErrors(cudaEventSynchronize(stagingDone));
        image->loadToHost(staging, chunk.startDown, chunk.startAcross, chunk.height, chunk.width);
        checkCudaErrors(cudaMemcpy2DAsync(buffer, pitch, staging, pitch, pitch, chunk.height,
            cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaEventRecord(stagingDone, stream));
    }
    else {
        // no page-locked memory: read to pageable memory (one buffer per thread), and copy;
        // a copy from pageable memory returns once the host buffer can be reused
        thread_local std::vector<char> host;
        host.resize(pitch * chunk.height);
        image->loadToHost(host.data(), chunk.startDown, chunk.startAcross, chunk.height, chunk.width);
        checkCudaErrors(cudaMemcpy2DAsync(buffer, pitch, host.data(), pitch, pitch, chunk.height,
            cudaMemcpyHostToDevice, stream));
    }
}

/**
 * A loader with the device buffers and the staging memory for the largest chunks of the reference
 * and secondary images
 */
cuAmpcorChunkLoader::cuAmpcorChunkLoader(
    const cuAmpcorChunkFootprint &referenceFootprint, DataType referenceDataType,
    SlcImage *referenceImage, const cuAmpcorChunkFootprint &secondaryFootprint,
    DataType secondaryDataType, SlcImage *secondaryImage, cudaStream_t stream_)
    : reference(referenceFootprint, referenceDataType, referenceImage),
      secondary(secondaryFootprint, secondaryDataType, secondaryImage),
      stream(stream_)
{
}

/**
 * Load the idxChunk-th chunk of both images (the copies to the device are queued on the stream)
 * @return the loaded chunks, valid until the next load
 */
const cuAmpcorChunk &cuAmpcorChunkLoader::load(int idxChunk)
{
    reference.load(idxChunk, chunk.reference, stream);
    secondary.load(idxChunk, chunk.secondary, stream);
    return chunk;
}

} // namespace
//end of file
