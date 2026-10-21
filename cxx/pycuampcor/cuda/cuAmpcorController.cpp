/**
 * @file  cuAmpcorController.cpp
 * @brief The controller running ampcor on a CUDA device
 */

#include "cuAmpcorController.h"
#include "cuAmpcorChunkLoader.h"
#include "cuAmpcorLayers.h"
#include "cuAmpcorProcessor.h"
#include "cuArrays.h"
#include "cuArraysCopy.h"
#include "cudaError.h"
#include "device.h"
#include "float2.h"
#include "logging.h"
#include "SlcImage.h"

namespace pycuampcor::cuda {

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

    /// allocate an array of height x width in the device memory
    template <typename T>
    static void newArray(std::unique_ptr<cuArrays<T>> &ptr, int height, int width)
    {
        ptr = std::make_unique<cuArrays<T>>(height, width);
        ptr->allocate();
    }

    /// extract the run images to the output images, and write them (and the gross offsets) to files
    void write(const cuAmpcorParameter *param, cudaStream_t stream)
    {
        cuArraysCopyExtract(offsetRun.get(), offset.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(snrRun.get(), snr.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(covRun.get(), cov.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(peakValueRun.get(), peakValue.get(), make_int2(0,0), stream);

        // the dense offsets, to the host
        offset->allocateHost();
        offset->copyToHost(stream);
        checkCudaErrors(cudaDeviceSynchronize());
        // the gross offsets of the windows
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

/// a worker: a CUDA stream, with its chunk loader and the processors of the layers
struct Worker {
    cudaStream_t stream;                                          ///< the stream of the worker
    std::unique_ptr<cuAmpcorChunkLoader> loader;                  ///< loads the chunks
    std::vector<std::unique_ptr<cuAmpcorProcessor>> processors;   ///< one per layer

    /// a worker with a new stream
    Worker() { checkCudaErrors(cudaStreamCreate(&stream)); }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    /// release the processors and the loader (their fft plans and copies use the stream), then the stream
    ~Worker()
    {
        processors.clear();
        loader.reset();
        checkCudaErrors(cudaStreamDestroy(stream));
    }
};

} // namespace

/**
 * The default number of windows in a chunk on the device
 */
void defaultWindowsInChunk(const cuAmpcorParameter &param, int &down, int &across)
{
    // windows in a chunk are processed in parallel; 2x the number of SMs keeps the GPU busy
    // (benchmarked on V100 and RTX PRO 6000 Blackwell), stacked down (SM/4) to share more rows
    const int sms = getSMCount(param.deviceID);
    down = std::max(sms / 4, 1);
    across = 8;
}

/**
 * Set up the parameters with the default number of windows in a chunk on the device
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
 * Each chunk is loaded once and processed by all layers. The device (deviceID) and the number
 * of streams (nStreams) are taken from the first layer.
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

    // select the device for this thread; all layers run on it
    const int deviceID = gpuDeviceInit(param->deviceID);
    for (auto *p : params)
        p->deviceID = deviceID;

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

    // the workers, one per stream, each with its loader and the processors of the layers
    const int nWorkers = param->nStreams;
    std::vector<std::unique_ptr<Worker>> workers(nWorkers);
    for (auto &worker : workers) {
        worker = std::make_unique<Worker>();
        worker->loader = std::make_unique<cuAmpcorChunkLoader>(
            footprints.first, param->referenceImageDataType, referenceImage.get(),
            footprints.second, param->secondaryImageDataType, secondaryImage.get(),
            worker->stream);
        for (int l = 0; l < nLayers; l++) {
            LayerImages &im = *images[l];
            worker->processors.push_back(cuAmpcorProcessor::create(params[l]->workflow, params[l],
                im.offsetRun.get(), im.snrRun.get(), im.covRun.get(), im.peakValueRun.get(), worker->stream));
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

    // assign the chunks to the streams in turn; the work of each chunk is queued on its stream
    const int message_interval = std::max(nChunksDown/10, 1);
    for (int k = 0; k < nChunks; k++) {
        const int i = k / nChunksAcross;
        const int j = k % nChunksAcross;
        if (j == 0 && i%message_interval == 0)
            logInfo("Processing chunks (", i+1, ", x) - (", std::min(nChunksDown, i+message_interval),
                ", x) out of ", nChunksDown);
        auto &worker = *workers[k % nWorkers];
        // load the chunk once, and process it for all layers
        const cuAmpcorChunk &chunk = worker.loader->load(k);
        for (auto &processor : worker.processors)
            processor->run(i, j, chunk);
    }

    // wait for all streams to finish
    checkCudaErrors(cudaDeviceSynchronize());

    // write the outputs of each layer
    for (int l = 0; l < nLayers; l++)
        images[l]->write(params[l], workers[0]->stream);
}

} // namespace
//end of file
