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
    // the device of the arrays (ArrayTraits.h)
    enum class Device : int;
    // the options of the parameters (Options.h)
    enum class Algorithm : int;
    enum class Workflow : int;
    enum class DerampMethod : int;
    enum class DerampAxis : int;
    enum class OversamplingMethod : int;
    enum class DataType : int;
    // the level of the messages (logging.h)
    enum class LogLevel : int;
    // the parameters (Parameter.h)
    struct Index2;
    class Parameter;
    // the images (SlcImage.h, Hdf5SlcImage.h, GdalSlcImage.h)
    class SlcImage;
    class MmapSlcImage;
    class Hdf5SlcImage;
    class GdalSlcImage;
    // the chunks of the layers processed together (Layers.h)
    struct ChunkFootprint;
    // the interfaces of the workers, for the arrays arrayT: the normalizer of the correlation
    // surfaces (Normalizer.h), the oversampler of the surfaces around the peaks (SurfaceOverSampler.h)
    template <template <typename> class arrayT>
    class Normalizer;
    template <template <typename> class arrayT>
    class SurfaceOverSampler;
    // the cross-correlators of the windows (Correlator.h): of T, by algorithm, on device (the
    // packages specialize it), and their interface for the algorithm chosen at run time
    template <Algorithm algorithm, typename T, Device device>
    class Correlator;
    template <template <typename> class arrayT>
    class WindowCorrelator;
    // the oversamplers of batches of 2D arrays of T, by method, on device (OverSampler.h; the
    // packages specialize it)
    template <OversamplingMethod method, typename T, Device device>
    class OverSampler;
    // the processors of a chunk, for each workflow, on the arrays arrayT (Processor*.h)
    template <template <typename> class arrayT>
    class Processor;
    template <template <typename> class arrayT>
    class ProcessorTwoPass;
    template <template <typename> class arrayT>
    class ProcessorOnePass;
}

//end of file
