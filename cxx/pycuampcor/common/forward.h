/**
 * @file  forward.h
 * @brief Forward declarations of the package
 */

#pragma once

#include "externals.h"

// the options of the parameters (cuAmpcorOptions.h)
namespace pycuampcor {
    enum class Algorithm : int;
    enum class Workflow : int;
    enum class DerampMethod : int;
    enum class DerampAxis : int;
    enum class OversamplingMethod : int;
    enum class DataType : int;
}

// the processing, compiled for each backend
namespace pycuampcor::PYCUAMPCOR_BACKEND {
    // the level of the messages (logging.h)
    enum class LogLevel : int;
    // the parameters (cuAmpcorParameter.h)
    class cuAmpcorParameter;
    // the images (SlcImage.h, Hdf5SlcImage.h)
    class SlcImage;
    class MmapSlcImage;
    class Hdf5SlcImage;
    // the chunks loaded from the images (cuAmpcorChunkLoader.h)
    struct cuAmpcorChunkFootprint;
    struct cuAmpcorLoadedChunk;
    struct cuAmpcorChunk;
    class cuAmpcorChunkLoader;
    // the processors of a chunk, for each workflow (cuAmpcorProcessor*.h)
    class cuAmpcorProcessor;
    class cuAmpcorProcessorTwoPass;
    class cuAmpcorProcessorOnePass;
    // the controller, running the processors over the chunks (cuAmpcorController.h)
    class cuAmpcorController;
}

// the backend: arrays, fft plans, oversamplers, correlators, normalizers, ...
#include "backend_forward.h"

//end of file
