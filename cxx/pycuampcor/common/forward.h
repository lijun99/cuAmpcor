/**
 * @file  forward.h
 * @brief Forward declarations of the shared code, in namespace pycuampcor
 *
 * The shared code (parameters, options, logging, image readers, layers) is compiled once and
 * used by the cpu and cuda packages, which have their own forward.h for their classes.
 */

#pragma once

#include "externals.h"

namespace pycuampcor {
    // the options of the parameters (cuAmpcorOptions.h)
    enum class Algorithm : int;
    enum class Workflow : int;
    enum class DerampMethod : int;
    enum class DerampAxis : int;
    enum class OversamplingMethod : int;
    enum class DataType : int;
    // the level of the messages (logging.h)
    enum class LogLevel : int;
    // the parameters (cuAmpcorParameter.h)
    struct Index2;
    class cuAmpcorParameter;
    // the images (SlcImage.h, Hdf5SlcImage.h)
    class SlcImage;
    class MmapSlcImage;
    class Hdf5SlcImage;
    // the chunks of the layers processed together (cuAmpcorLayers.h)
    struct cuAmpcorChunkFootprint;
}

//end of file
