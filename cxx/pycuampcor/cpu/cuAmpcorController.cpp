/**
 * @file  cuAmpcorController.cpp
 * @brief The controller running ampcor on the CPU
 */

#include "cuAmpcorController.h"
#include "cuAmpcorChunkLoader.h"
#include "cuAmpcorLayers.h"
#include "cuAmpcorProcessor.h"
#include "cuArrays.h"
#include "cuArraysCopy.h"
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
    std::unique_ptr<cuArrays<real2_type>> offsetRun, offset;        ///< the dense offsets
    std::unique_ptr<cuArrays<real_type>> snrRun, snr;               ///< the SNR
    std::unique_ptr<cuArrays<real3_type>> covRun, cov;              ///< the covariance
    std::unique_ptr<cuArrays<real_type>> peakValueRun, peakValue;   ///< the correlation peak

    /// allocate the run and output images of a layer
    explicit LayerImages(const cuAmpcorParameter *param)
    {
        // the run images cover whole chunks: they may be larger than the output images
        const int nWindowsDownRun = param->numberChunkDown * param->windowsDownInChunk;
        const int nWindowsAcrossRun = param->numberChunkAcross * param->windowsAcrossInChunk;
        newArray(offsetRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(snrRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(covRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(peakValueRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(offset, param->numberWindowDown, param->numberWindowAcross);
        newArray(snr, param->numberWindowDown, param->numberWindowAcross);
        newArray(cov, param->numberWindowDown, param->numberWindowAcross);
        newArray(peakValue, param->numberWindowDown, param->numberWindowAcross);
    }

    /// allocate an array of height x width in the work memory
    template <typename T>
    static void newArray(std::unique_ptr<cuArrays<T>> &ptr, int height, int width)
    {
        ptr = std::make_unique<cuArrays<T>>(height, width);
        ptr->allocate();
    }

    /// extract the run images to the output images, and write them (and the gross offsets) to files
    void write(const cuAmpcorParameter *param)
    {
        const stream_t stream{};
        cuArraysCopyExtract(offsetRun.get(), offset.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(snrRun.get(), snr.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(covRun.get(), cov.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(peakValueRun.get(), peakValue.get(), make_int2(0,0), stream);

        // the gross offsets of the windows
        offset->allocateHost();
        offset->copyToHost(stream);
        cuArrays<real2_type> grossOffset(param->numberWindowDown, param->numberWindowAcross);
        grossOffset.allocateHost();
        for(int i=0; i< param->numberWindows; i++)
            grossOffset.hostData[i] = make_real2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);
        // the dense offsets, with the gross offsets added if they are merged
        if (param->mergeGrossOffset) {
            for(int i=0; i< param->numberWindows; i++)
                offset->hostData[i] += grossOffset.hostData[i];
        }
        offset->outputHostToFile(param->offsetImageName);
        grossOffset.outputHostToFile(param->grossOffsetImageName);

        // the statistics
        snr->outputToFile(param->snrImageName, stream);
        cov->outputToFile(param->covImageName, stream);
        peakValue->outputToFile(param->peakValueImageName, stream);
    }
};

/// a worker (an OpenMP thread), with its chunk loader and the processors of the layers
struct Worker {
    std::unique_ptr<cuAmpcorChunkLoader> loader;                  ///< loads the chunks
    std::vector<std::unique_ptr<cuAmpcorProcessor>> processors;   ///< one per layer
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
    const cuAmpcorParameter *param = params.front();
    const int nLayers = params.size();

    const int nWorkers = numberOfThreads(*param);
    logInfo("Using ", nWorkers, " CPU thread(s) ...");

    // the reference and secondary images, with the largest buffer of the layers
    int mmapSizeInGB = 0;
    for (const auto *p : params)
        mmapSizeInGB = std::max(mmapSizeInGB, p->mmapSizeInGB);
    logInfo("Opening reference image ", param->referenceImageName, "...");
    auto referenceImage = SlcImage::open(param->referenceImageName, param->referenceImageReader,
        param->referenceImageHeight, param->referenceImageWidth,
        pixelSize(param->referenceImageDataType), mmapSizeInGB);
    logInfo("Opening secondary image ", param->secondaryImageName, "...");
    auto secondaryImage = SlcImage::open(param->secondaryImageName, param->secondaryImageReader,
        param->secondaryImageHeight, param->secondaryImageWidth,
        pixelSize(param->secondaryImageDataType), mmapSizeInGB);

    // the output images of each layer
    std::vector<std::unique_ptr<LayerImages>> images;
    for (const auto *p : params)
        images.push_back(std::make_unique<LayerImages>(p));

    // the workers, one per thread, each with its loader and the processors of the layers
    std::vector<std::unique_ptr<Worker>> workers(nWorkers);
    for (auto &worker : workers) {
        worker = std::make_unique<Worker>();
        worker->loader = std::make_unique<cuAmpcorChunkLoader>(
            footprints.first, param->referenceImageDataType, referenceImage.get(),
            footprints.second, param->secondaryImageDataType, secondaryImage.get());
        for (int l = 0; l < nLayers; l++) {
            LayerImages &im = *images[l];
            worker->processors.push_back(cuAmpcorProcessor::create(params[l]->workflow, params[l],
                im.offsetRun.get(), im.snrRun.get(), im.covRun.get(), im.peakValueRun.get(), stream_t{}));
        }
    }

    const int nChunksDown = param->numberChunkDown;
    const int nChunksAcross = param->numberChunkAcross;
    const int nChunks = nChunksDown*nChunksAcross;
    logInfo("Total number of windows (azimuth x range):  ",
        param->numberWindowDown, " x ", param->numberWindowAcross);
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
        images[l]->write(params[l]);
}

} // namespace
//end of file
