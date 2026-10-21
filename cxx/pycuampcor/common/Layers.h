/**
 * @file  Layers.h
 * @brief Layers: several parameter sets processed together, sharing the loaded image chunks
 *
 * A layer is a set of parameters (e.g., template/search window sizes) applied to the same
 * images and the same chunk partition (numberWindow*, windows*InChunk). Each chunk is loaded
 * once, covering the windows of all layers (its footprint), and each layer's processor then
 * copies its windows from the loaded chunk.
 */

#pragma once

#include "forward.h"

namespace pycuampcor {

/// the area of each chunk of an image to load, covering the windows of all layers
struct ChunkFootprint {
    std::vector<int> startDown;    ///< first line of each chunk
    std::vector<int> startAcross;  ///< first column of each chunk
    std::vector<int> height;       ///< height of each chunk; 0 if the chunk is outside the image
    std::vector<int> width;        ///< width of each chunk; 0 if the chunk is outside the image
    int maxHeight = 0;             ///< the largest chunk height
    int maxWidth = 0;              ///< the largest chunk width
};

/// check that the layers can be processed together: set up (with their starting pixels), with
/// the same images, numbers of windows and windows in a chunk, and distinct output files
/// @throw std::invalid_argument if not
void checkLayers(const std::vector<const Parameter *> &layers);

/// the footprints of the chunks of the reference and secondary images, covering all layers
/// @param layers the parameters of the layers, checked by checkLayers
/// @return the footprints of the reference and secondary chunks
std::pair<ChunkFootprint, ChunkFootprint>
    chunkFootprints(const std::vector<const Parameter *> &layers);

} // namespace
//end of file
