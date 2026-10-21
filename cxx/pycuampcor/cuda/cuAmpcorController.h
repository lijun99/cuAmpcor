/**
 * @file  cuAmpcorController.h
 * @brief The controller running ampcor on a CUDA device
 *
 * The controller divides the windows into chunks (batches of windows, processed in parallel on
 * the device), and assigns the chunks to CUDA streams in turn: each stream loads a chunk of the
 * images and runs the processor of each layer on it, so that the copies of a stream overlap with
 * the kernels of the others. The outputs are written once all chunks are processed.
 */

#pragma once

#include "forward.h"
#include "cuAmpcorParameter.h"

namespace pycuampcor::cuda {

/// the controller running ampcor on a CUDA device, with its parameters
class cuAmpcorController {
public:
    /// the parameters
    std::unique_ptr<cuAmpcorParameter> param = std::make_unique<cuAmpcorParameter>();

    /// run ampcor with the parameters (set up, with the starting pixels), and write the outputs
    void runAmpcor();

    /// run several layers (parameter sets) sharing the images and the chunk partition, loading
    /// each chunk once for all layers; the device and the number of streams are taken from the
    /// first layer
    static void runAmpcorLayers(const std::vector<cuAmpcorController *> &layers);

    /// whether the computations and the outputs are in double precision
    bool isDoublePrecision();
};

/// the default number of windows in a chunk on the device (deviceID): 2x the number of SMs, as
/// (SM/4) x 8 windows (down x across)
/// @param param the parameters
/// @param[out] down, across the number of windows in a chunk, down and across
void defaultWindowsInChunk(const cuAmpcorParameter &param, int &down, int &across);

/// set up the parameters (cuAmpcorParameter::setupParameters), with the default number of
/// windows in a chunk on the device
void setupParameters(cuAmpcorParameter &param);

} // namespace
//end of file
