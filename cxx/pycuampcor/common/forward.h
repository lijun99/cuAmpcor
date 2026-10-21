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
    // a view of a batch of 2D arrays (ArraysView.h)
    template <typename T>
    class ArraysView;
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
    // the interfaces of the workers, for the arrays arrayT: the normalizer of the correlation
    // surfaces (cuNormalizer.h), the oversampler of the surfaces around the peaks (cuSurfaceOverSampler.h)
    template <template <typename> class arrayT>
    class cuNormalizer;
    template <template <typename> class arrayT>
    class cuSurfaceOverSampler;
    // the processors of a chunk, for each workflow, on the arrays arrayT (cuAmpcorProcessor*.h)
    template <template <typename> class arrayT>
    class cuAmpcorProcessor;
    template <template <typename> class arrayT>
    class cuAmpcorProcessorTwoPass;
    template <template <typename> class arrayT>
    class cuAmpcorProcessorOnePass;
}

//end of file
