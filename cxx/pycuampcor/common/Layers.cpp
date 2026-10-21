/**
 * @file  Layers.cpp
 * @brief Layers: several parameter sets processed together, sharing the loaded image chunks
 */

#include "Layers.h"
#include "Parameter.h"

#include <climits>
#include <set>

namespace pycuampcor {

namespace {

/**
 * The union of the chunks of all layers, for one image
 * @param layers the parameters of the layers
 * @param reference whether for the reference image (or the secondary image)
 */
ChunkFootprint unionFootprint(const std::vector<const Parameter *> &layers, bool reference)
{
    const int nChunks = layers.front()->numberChunks;
    ChunkFootprint fp;
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

/**
 * Check that the layers can be processed together (see Layers.h)
 * @param params the parameters of the layers
 */
void checkLayers(const std::vector<const Parameter *> &params)
{
    if (params.empty())
        throw std::invalid_argument("No layers are given");
    const Parameter *p0 = params.front();
    std::set<std::string> outputs;
    for (const auto *p : params) {
        p->checkReadyToRun();
        if (p->referenceImageName != p0->referenceImageName
            || p->referenceImageReader != p0->referenceImageReader
            || p->referenceImageHeight != p0->referenceImageHeight
            || p->referenceImageWidth != p0->referenceImageWidth
            || p->referenceImageDataType != p0->referenceImageDataType
            || p->secondaryImageName != p0->secondaryImageName
            || p->secondaryImageReader != p0->secondaryImageReader
            || p->secondaryImageHeight != p0->secondaryImageHeight
            || p->secondaryImageWidth != p0->secondaryImageWidth
            || p->secondaryImageDataType != p0->secondaryImageDataType)
            throw std::invalid_argument("All layers must use the same reference and secondary images");
        if (p->numberWindowDown != p0->numberWindowDown
            || p->numberWindowAcross != p0->numberWindowAcross
            || p->windowsDownInChunk != p0->windowsDownInChunk
            || p->windowsAcrossInChunk != p0->windowsAcrossInChunk)
            throw std::invalid_argument("All layers must have the same numbers of windows and windows in a chunk");
        for (const auto &name : {p->offsetImageName, p->grossOffsetImageName, p->snrImageName,
                                 p->covImageName, p->peakValueImageName})
            if (!outputs.insert(name).second)
                throw std::invalid_argument("Output file " + name + " is used more than once");
    }
}

/**
 * The footprints of the chunks of the reference and secondary images (see Layers.h)
 * @param layers the parameters of the layers
 */
std::pair<ChunkFootprint, ChunkFootprint>
chunkFootprints(const std::vector<const Parameter *> &layers)
{
    checkLayers(layers);
    return {unionFootprint(layers, true), unionFootprint(layers, false)};
}

} // namespace
//end of file
