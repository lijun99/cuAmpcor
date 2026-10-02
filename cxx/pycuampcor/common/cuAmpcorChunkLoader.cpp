#include "cuAmpcorChunkLoader.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

namespace {

// the union of the chunks of all layers, for one image
cuAmpcorChunkFootprint unionFootprint(const std::vector<const cuAmpcorParameter *> &layers, bool reference)
{
    const int nChunks = layers.front()->numberChunks;
    cuAmpcorChunkFootprint fp;
    for (auto *v : {&fp.startDown, &fp.startAcross, &fp.height, &fp.width})
        v->assign(nChunks, 0);

    for (int c = 0; c < nChunks; c++) {
        int sD = INT_MAX, sA = INT_MAX, eD = INT_MIN, eA = INT_MIN;
        for (const auto *p : layers) {
            const int h = reference ? p->referenceChunkHeight[c] : p->secondaryChunkHeight[c];
            const int w = reference ? p->referenceChunkWidth[c] : p->secondaryChunkWidth[c];
            // skip chunks entirely outside the image
            if (h == 0 || w == 0) continue;
            const int d = reference ? p->referenceChunkStartPixelDown[c] : p->secondaryChunkStartPixelDown[c];
            const int a = reference ? p->referenceChunkStartPixelAcross[c] : p->secondaryChunkStartPixelAcross[c];
            sD = std::min(sD, d);
            sA = std::min(sA, a);
            eD = std::max(eD, d + h);
            eA = std::max(eA, a + w);
        }
        if (eD > sD && eA > sA) {
            fp.startDown[c] = sD;
            fp.startAcross[c] = sA;
            fp.height[c] = eD - sD;
            fp.width[c] = eA - sA;
            fp.maxHeight = std::max(fp.maxHeight, fp.height[c]);
            fp.maxWidth = std::max(fp.maxWidth, fp.width[c]);
        }
    }
    return fp;
}

} // namespace

std::pair<cuAmpcorChunkFootprint, cuAmpcorChunkFootprint>
cuAmpcorChunkLoader::footprints(const std::vector<const cuAmpcorParameter *> &layers)
{
    if (layers.empty())
        throw std::invalid_argument("No layers are given");
    for (const auto *p : layers) {
        p->checkReadyToRun();
        if (p->numberChunks != layers.front()->numberChunks)
            throw std::invalid_argument("All layers must share the same chunk partition");
    }
    return {unionFootprint(layers, true), unionFootprint(layers, false)};
}

cuAmpcorChunkLoader::Source::Source(const cuAmpcorChunkFootprint &footprint_, int dataType, SlcImage *image_)
    : footprint(footprint_), image(image_)
{
    // allocate the buffer once for the largest chunk
    // (allocating/freeing device memory per chunk synchronizes the device and stalls the other streams)
    const int height = std::max(footprint.maxHeight, 1);
    const int width = std::max(footprint.maxWidth, 1);
    if (dataType == 2) {
        complexBuffer = std::make_unique<cuArrays<image_complex_type>>(height, width);
        complexBuffer->allocate();
    }
    else {
        realBuffer = std::make_unique<cuArrays<image_real_type>>(height, width);
        realBuffer->allocate();
    }
    // page-locked host memory for the largest chunk, so that copies to the device are asynchronous
    staging = backendAllocStaging(static_cast<size_t>(height) * width * image->pixelSize());
    stagingDone = backendCreateEvent();
}

cuAmpcorChunkLoader::Source::~Source()
{
    // the last copy from the staging memory must be done before it is freed
    backendWaitEvent(stagingDone);
    backendDestroyEvent(stagingDone);
    backendFreeStaging(staging);
}

void cuAmpcorChunkLoader::Source::load(int idxChunk, cuAmpcorLoadedChunk &chunk, stream_t stream)
{
    chunk.complexData = complexBuffer.get();
    chunk.realData = realBuffer.get();
    chunk.startDown = footprint.startDown[idxChunk];
    chunk.startAcross = footprint.startAcross[idxChunk];
    chunk.height = footprint.height[idxChunk];
    chunk.width = footprint.width[idxChunk];
    if (chunk.empty()) return;
    void *buffer = complexBuffer ? (void *)complexBuffer->devData : (void *)realBuffer->devData;
    if (staging) {
        // reuse the staging memory once its previous copy is done, then copy asynchronously,
        // so the worker can go on queuing its work (and the other workers can load) during the copy
        backendWaitEvent(stagingDone);
        image->loadToHost(staging, chunk.startDown, chunk.startAcross, chunk.height, chunk.width);
        const size_t pitch = static_cast<size_t>(chunk.width) * image->pixelSize();
        backendCopyFromHost2D(buffer, pitch, staging, pitch, pitch, chunk.height, stream);
        backendRecordEvent(stagingDone, stream);
    }
    else if (backendWorkInHostMemory) {
        // the work memory is host memory (CPU backend)
        image->loadToHost(buffer, chunk.startDown, chunk.startAcross, chunk.height, chunk.width);
    }
    else {
        // no page-locked memory: copy from pageable memory
        image->loadToDevice(buffer, chunk.startDown, chunk.startAcross, chunk.height, chunk.width, stream);
    }
}

cuAmpcorChunkLoader::cuAmpcorChunkLoader(
    const cuAmpcorChunkFootprint &referenceFootprint, int referenceDataType, SlcImage *referenceImage,
    const cuAmpcorChunkFootprint &secondaryFootprint, int secondaryDataType, SlcImage *secondaryImage,
    stream_t stream_)
    : reference(referenceFootprint, referenceDataType, referenceImage),
      secondary(secondaryFootprint, secondaryDataType, secondaryImage),
      stream(stream_)
{
}

const cuAmpcorChunk &cuAmpcorChunkLoader::load(int idxChunk)
{
    reference.load(idxChunk, chunk.reference, stream);
    secondary.load(idxChunk, chunk.secondary, stream);
    return chunk;
}

} // namespace
