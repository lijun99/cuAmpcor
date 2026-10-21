/**
 * @file  cuAmpcorController.h
 * @brief The controller for running cuAmcor
 *
 * cuAmpController is the main processor, also interface to python
 * It determines the total number of windows, the starting pixels for each window.
 * It then divides windows into chunks (batches), and creates cuAmpcorChunk instances
 *  to process each chunk.
 * A chunk includes multiple windows, to maximize the use of GPU cores.
 * Different cuAmpcorChunk processors use different cuda streams, to overlap
 *  the kernel execution with data copying.
 */

#pragma once

// dependencies
#include "cuAmpcorParameter.h"
#include <memory>
#include <vector>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

class cuAmpcorController {
public:
    std::unique_ptr<cuAmpcorParameter> param = std::make_unique<cuAmpcorParameter>();  ///< the parameter set
    // run interface
    void runAmpcor();
    // run several layers (parameter sets) sharing the images and the chunk partition,
    // loading each chunk once for all layers
    static void runAmpcorLayers(const std::vector<cuAmpcorController *> &layers);
    bool isDoublePrecision();
};
} // namespace
// end of file
