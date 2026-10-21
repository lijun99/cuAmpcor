/**
 * @file  cuAmpcorController.cpp
 * @brief The controller running ampcor on the CPU
 */

#include "cuAmpcorController.h"
#include "cuAmpcorChunkLoader.h"
#include "cuAmpcorLayers.h"
#include "api.h"
#include "arrayTraits.h"
#include "cuAmpcorProcessor.h"
#include "Arrays.h"
#include "ArraysFile.h"
#include "ArraysCopy.h"
#include "float2.h"
#include "logging.h"
#include "SlcImage.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pycuampcor::cpu {

namespace {

/// the output images of a layer; the run images, written by the processors, are padded to
/// whole chunks, and extracted to the output images at the end
struct LayerImages {
    array_type<real2_type> offsetRun, offset;        ///< the dense offsets
    array_type<real_type> snrRun, snr;               ///< the SNR
    array_type<real3_type> covRun, cov;              ///< the covariance
    array_type<real_type> peakValueRun, peakValue;   ///< the correlation peak

    /// allocate the run and output images of a layer; the run images cover whole chunks: they may
    /// be larger than the output images
    explicit LayerImages(const cuAmpcorParameter &param)
        : offsetRun(runHeight(param), runWidth(param)), offset(param.numberWindowDown, param.numberWindowAcross),
          snrRun(runHeight(param), runWidth(param)), snr(param.numberWindowDown, param.numberWindowAcross),
          covRun(runHeight(param), runWidth(param)), cov(param.numberWindowDown, param.numberWindowAcross),
          peakValueRun(runHeight(param), runWidth(param)), peakValue(param.numberWindowDown, param.numberWindowAcross)
    {}

    /// the height of the run images: the windows down in whole chunks
    static int runHeight(const cuAmpcorParameter &param)
    {
        return param.numberChunkDown * param.windowsDownInChunk;
    }
    /// the width of the run images: the windows across in whole chunks
    static int runWidth(const cuAmpcorParameter &param)
    {
        return param.numberChunkAcross * param.windowsAcrossInChunk;
    }

    /// extract the run images to the output images, and write them (and the gross offsets) to files
    void write(const cuAmpcorParameter &param)
    {
        const stream_t stream{};
        cuArraysCopyExtract(offsetRun, offset, make_int2(0,0), stream);
        cuArraysCopyExtract(snrRun, snr, make_int2(0,0), stream);
        cuArraysCopyExtract(covRun, cov, make_int2(0,0), stream);
        cuArraysCopyExtract(peakValueRun, peakValue, make_int2(0,0), stream);

        // the dense offsets, to the host
        std::vector<real2_type> offsetHost = offset.toHost(stream);
        // the gross offsets of the windows
        std::vector<real2_type> grossOffset(offsetHost.size());
        for(int i=0; i< param.numberWindows; i++)
            grossOffset[i] = make_real2(param.grossOffsetDown[i], param.grossOffsetAcross[i]);
        // the dense offsets, with the gross offsets added if they are merged
        if (param.mergeGrossOffset) {
            for(int i=0; i< param.numberWindows; i++)
                offsetHost[i] += grossOffset[i];
        }
        writeArraysToFile(param.offsetImageName, ArraysView<const real2_type>(offsetHost.data(),
            offset.height(), offset.width()));
        writeArraysToFile(param.grossOffsetImageName, ArraysView<const real2_type>(grossOffset.data(),
            offset.height(), offset.width()));

        // the statistics
        snr.outputToFile(param.snrImageName, stream);
        cov.outputToFile(param.covImageName, stream);
        peakValue.outputToFile(param.peakValueImageName, stream);
    }
};

/// a worker (an OpenMP thread), with its chunk loader and the processors of the layers
struct Worker {
    std::unique_ptr<cuAmpcorChunkLoader> loader;                  ///< loads the chunks
    std::vector<std::unique_ptr<processor_t>> processors;   ///< one per layer
};

} // namespace

/**
 * The default number of windows in a chunk on the CPU
 */
void defaultWindowsInChunk(const cuAmpcorParameter &, int &down, int &across)
{
    // each thread processes a chunk, its windows one after another; one window per chunk
    // keeps the work in cache and balances the load among threads
    down = 1;
    across = 1;
}

/**
 * Set up the parameters with the default number of windows in a chunk on the CPU
 */
void setupParameters(cuAmpcorParameter &param)
{
    // the default is needed (and, for CUDA, the device queried) only if a number is automatic (0)
    int down = 0, across = 0;
    if (param.numberWindowDownInChunk <= 0 || param.numberWindowAcrossInChunk <= 0)
        defaultWindowsInChunk(param, down, across);
    param.setupParameters(down, across);
}

/**
 * The number of threads processing the chunks
 */
int numberOfThreads(const cuAmpcorParameter &param)
{
    if (param.nThreads > 0)
        return param.nThreads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * Whether the computations and the outputs are in double precision
 */
bool cuAmpcorController::isDoublePrecision()
{
#ifdef CUAMPCOR_DOUBLE
    return true;
#else
    return false;
#endif
}

/**
 * Run ampcor with the parameters, and write the outputs
 */
void cuAmpcorController::runAmpcor()
{
    runAmpcorLayers({this});
}

/**
 * Run several layers (parameter sets) sharing the images and the chunk partition
 * Each chunk is loaded once and processed by all layers. The number of threads (nThreads) is
 * taken from the first layer.
 */
void cuAmpcorController::runAmpcorLayers(const std::vector<cuAmpcorController *> &layers)
{
    std::vector<cuAmpcorParameter *> params;
    for (auto *layer : layers)
        params.push_back(layer->param.get());
    // check whether the parameters are set up and compatible, and get the chunks to load
    const auto footprints = chunkFootprints({params.begin(), params.end()});
    const cuAmpcorParameter &param = *params.front();
    const int nLayers = params.size();

    const int nWorkers = numberOfThreads(param);
    logInfo("Using ", nWorkers, " CPU thread(s) ...");

    // the reference and secondary images, with the largest buffer of the layers
    int mmapSizeInGB = 0;
    for (const auto *p : params)
        mmapSizeInGB = std::max(mmapSizeInGB, p->mmapSizeInGB);
    logInfo("Opening reference image ", param.referenceImageName, "...");
    auto referenceImage = SlcImage::open(param.referenceImageName, param.referenceImageReader,
        param.referenceImageHeight, param.referenceImageWidth,
        pixelSize(param.referenceImageDataType), mmapSizeInGB);
    logInfo("Opening secondary image ", param.secondaryImageName, "...");
    auto secondaryImage = SlcImage::open(param.secondaryImageName, param.secondaryImageReader,
        param.secondaryImageHeight, param.secondaryImageWidth,
        pixelSize(param.secondaryImageDataType), mmapSizeInGB);

    // the output images of each layer
    std::vector<std::unique_ptr<LayerImages>> images;
    for (const auto *p : params)
        images.push_back(std::make_unique<LayerImages>(*p));

    // the workers, one per thread, each with its loader and the processors of the layers
    std::vector<std::unique_ptr<Worker>> workers(nWorkers);
    for (auto &worker : workers) {
        worker = std::make_unique<Worker>();
        worker->loader = std::make_unique<cuAmpcorChunkLoader>(
            footprints.first, param.referenceImageDataType, *referenceImage,
            footprints.second, param.secondaryImageDataType, *secondaryImage);
        for (int l = 0; l < nLayers; l++) {
            LayerImages &im = *images[l];
            worker->processors.push_back(processor_t::create(params[l]->workflow, *params[l],
                im.offsetRun, im.snrRun, im.covRun, im.peakValueRun, stream_t{}));
        }
    }

    const int nChunksDown = param.numberChunkDown;
    const int nChunksAcross = param.numberChunkAcross;
    const int nChunks = nChunksDown*nChunksAcross;
    logInfo("Total number of windows (azimuth x range):  ",
        param.numberWindowDown, " x ", param.numberWindowAcross);
    logInfo("to be processed in the number of chunks: ", nChunksDown, " x ", nChunksAcross);
    if (nLayers > 1)
        logInfo("for ", nLayers, " layers sharing the loaded chunks");

    // process the chunks in parallel, each thread with its worker
    // (exceptions can't propagate out of an OpenMP region: keep the first one, and rethrow it after)
    const int message_interval = std::max(nChunksDown/10, 1);
    std::exception_ptr error;
    std::atomic<bool> failed(false);
    #pragma omp parallel for schedule(dynamic) num_threads(nWorkers)
    for (int k = 0; k < nChunks; k++) {
        if (failed) continue;
        const int i = k / nChunksAcross;
        const int j = k % nChunksAcross;
        if (j == 0 && i%message_interval == 0)
            logInfo("Processing chunks (", i+1, ", x) - (", std::min(nChunksDown, i+message_interval),
                ", x) out of ", nChunksDown);
        try {
#ifdef _OPENMP
            auto &worker = *workers[omp_get_thread_num()];
#else
            auto &worker = *workers[0];
#endif
            // load the chunk once, and process it for all layers
            const cuAmpcorChunk &chunk = worker.loader->load(k);
            for (auto &processor : worker.processors)
                processor->run(i, j, chunk);
        }
        catch (...) {
            #pragma omp critical
            {
                if (!failed) error = std::current_exception();
                failed = true;
            }
        }
    }
    if (error) std::rethrow_exception(error);

    // write the outputs of each layer
    for (int l = 0; l < nLayers; l++)
        images[l]->write(*params[l]);
}

} // namespace
//end of file
