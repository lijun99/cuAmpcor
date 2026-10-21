/**
 * @file  cuController.h
 * @brief The controller running ampcor on a CUDA device
 *
 * The controller divides the windows into chunks (batches of windows, processed in parallel on
 * the device), and assigns the chunks to CUDA streams in turn: each stream loads a chunk of the
 * images and runs the processor of each layer on it, so that the copies of a stream overlap with
 * the kernels of the others. The outputs are written once all chunks are processed.
 */

#pragma once

#include "forward.h"
#include "Parameter.h"

namespace pycuampcor::cuda {

/// the controller running ampcor on a CUDA device, with its parameters
class Controller {
public:
    /// the parameters
    std::unique_ptr<Parameter> param = std::make_unique<Parameter>();

    /// run ampcor with the parameters (set up, with the starting pixels), and write the outputs
    void runAmpcor();

    /// run several layers (parameter sets) sharing the images and the chunk partition, loading
    /// each chunk once for all layers; the device and the number of streams are taken from the
    /// first layer
    static void runAmpcorLayers(const std::vector<Controller *> &layers);

    /// whether the computations and the outputs are in double precision
    bool isDoublePrecision();
};

/// the default number of windows in a chunk on the device (deviceID): 2x the number of SMs, as
/// (SM/4) x 8 windows (down x across)
/// @param param the parameters
/// @param[out] down,across the number of windows in a chunk, down and across
void defaultWindowsInChunk(const Parameter &param, int &down, int &across);

/// set up the parameters (Parameter::setupParameters), with the default number of
/// windows in a chunk on the device
void setupParameters(Parameter &param);

} // namespace
//end of file
