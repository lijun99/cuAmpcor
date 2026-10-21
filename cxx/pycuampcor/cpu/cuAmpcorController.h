/**
 * @file  cuAmpcorController.h
 * @brief The controller running ampcor on the CPU
 *
 * The controller divides the windows into chunks (batches of windows), and processes the
 * chunks in parallel with OpenMP threads: each thread loads a chunk of the images and runs the
 * processor of each layer on it. The outputs are written once all chunks are processed.
 */

#pragma once

#include "forward.h"
#include "cuAmpcorParameter.h"

namespace pycuampcor::cpu {

/// the controller running ampcor on the CPU, with its parameters
class cuAmpcorController {
public:
    /// the parameters
    std::unique_ptr<cuAmpcorParameter> param = std::make_unique<cuAmpcorParameter>();

    /// run ampcor with the parameters (set up, with the starting pixels), and write the outputs
    void runAmpcor();

    /// run several layers (parameter sets) sharing the images and the chunk partition, loading
    /// each chunk once for all layers; the number of threads is taken from the first layer
    static void runAmpcorLayers(const std::vector<cuAmpcorController *> &layers);

    /// whether the computations and the outputs are in double precision
    bool isDoublePrecision();
};

/// the default number of windows in a chunk on the CPU: one window per chunk, as each thread
/// processes a chunk (its windows one after another)
/// @param param the parameters
/// @param[out] down,across the number of windows in a chunk, down and across
void defaultWindowsInChunk(const cuAmpcorParameter &param, int &down, int &across);

/// set up the parameters (cuAmpcorParameter::setupParameters), with the default number of
/// windows in a chunk on the CPU
void setupParameters(cuAmpcorParameter &param);

/// the number of threads processing the chunks: nThreads, or the OpenMP default if 0
int numberOfThreads(const cuAmpcorParameter &param);

} // namespace
//end of file
