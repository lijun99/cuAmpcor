/**
 * @file cuAmpcorController.cu
 * @brief Implementations of cuAmpcorController
 */

// my declaration
#include "cuAmpcorController.h"

// dependencies
#include "SlcImage.h"
#include "cuArrays.h"
#include "cuAmpcorProcessor.h"
#include "cuAmpcorUtil.h"
#include "backend.h"
#include "cuAmpcorChunkLoader.h"
#include <algorithm>
#include <atomic>
#include <exception>
#include <cstdio>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

// constructor
cuAmpcorController::cuAmpcorController()
{
    // create a new set of parameters
    param = new cuAmpcorParameter();
}

// destructor
cuAmpcorController::~cuAmpcorController()
{
    delete param;
}

bool cuAmpcorController::isDoublePrecision()
{
#ifdef CUAMPCOR_DOUBLE
    return true;
#else
    return false;
#endif
}
namespace {

// output images of a layer; the run images are padded to whole chunks
struct LayerImages {
    std::unique_ptr<cuArrays<real2_type>> offsetRun, offset;
    std::unique_ptr<cuArrays<real_type>> snrRun, snr;
    std::unique_ptr<cuArrays<real3_type>> covRun, cov;
    std::unique_ptr<cuArrays<real_type>> peakValueRun, peakValue;

    explicit LayerImages(const cuAmpcorParameter *param)
    {
        // nWindowsDownRun is defined as numberChunk * numberWindowInChunk
        // It may be bigger than the actual number of windows
        const int nWindowsDownRun = param->numberChunkDown * param->numberWindowDownInChunk;
        const int nWindowsAcrossRun = param->numberChunkAcross * param->numberWindowAcrossInChunk;
        newArray(offsetRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(snrRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(covRun, nWindowsDownRun, nWindowsAcrossRun);
        newArray(peakValueRun, nWindowsDownRun, nWindowsAcrossRun);
        // output images: offset fields, SNR, variance, and correlation surface peak value
        newArray(offset, param->numberWindowDown, param->numberWindowAcross);
        newArray(snr, param->numberWindowDown, param->numberWindowAcross);
        newArray(cov, param->numberWindowDown, param->numberWindowAcross);
        newArray(peakValue, param->numberWindowDown, param->numberWindowAcross);
    }

    // allocate an array in (device) memory
    template <typename T>
    static void newArray(std::unique_ptr<cuArrays<T>> &ptr, int height, int width)
    {
        ptr = std::make_unique<cuArrays<T>>(height, width);
        ptr->allocate();
    }

    // extract the run images to the output images, and save them to files
    void write(const cuAmpcorParameter *param, stream_t stream)
    {
        cuArraysCopyExtract(offsetRun.get(), offset.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(snrRun.get(), snr.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(covRun.get(), cov.get(), make_int2(0,0), stream);
        cuArraysCopyExtract(peakValueRun.get(), peakValue.get(), make_int2(0,0), stream);

        /* save the offsets and gross offsets */
        // copy the offset to host
        offset->allocateHost();
        offset->copyToHost(stream);
        backendSynchronize();
        // construct the gross offset
        cuArrays<real2_type> grossOffset(param->numberWindowDown, param->numberWindowAcross);
        grossOffset.allocateHost();
        for(int i=0; i< param->numberWindows; i++)
            grossOffset.hostData[i] = make_real2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);

        // check whether to merge gross offset
        if (param->mergeGrossOffset)
        {
            // if merge, add the gross offsets to offset
            for(int i=0; i< param->numberWindows; i++)
                offset->hostData[i] += grossOffset.hostData[i];
        }
        // output both offset and gross offset
        offset->outputHostToFile(param->offsetImageName);
        grossOffset.outputHostToFile(param->grossOffsetImageName);

        // save the snr/cov images
        snr->outputToFile(param->snrImageName, stream);
        cov->outputToFile(param->covImageName, stream);
        peakValue->outputToFile(param->peakValueImageName, stream);
    }
};

// check that the layers can share the loaded chunks
void checkLayers(const std::vector<cuAmpcorParameter *> &params)
{
    if (params.empty())
        throw std::invalid_argument("No layers are given");
    const cuAmpcorParameter *p0 = params.front();
    std::set<std::string> outputs;
    for (const auto *p : params) {
        p->checkReadyToRun();
        if (p->referenceImageName != p0->referenceImageName
            || p->referenceImageReader != p0->referenceImageReader
            || p->referenceImageHeight != p0->referenceImageHeight
            || p->referenceImageWidth != p0->referenceImageWidth
            || p->referenceImageDataType != p0->referenceImageDataType
            || p->secondaryImageName != p0->secondaryImageName
            || p->secondaryImageReader != p0->secondaryImageReader
            || p->secondaryImageHeight != p0->secondaryImageHeight
            || p->secondaryImageWidth != p0->secondaryImageWidth
            || p->secondaryImageDataType != p0->secondaryImageDataType)
            throw std::invalid_argument("All layers must use the same reference and secondary images");
        if (p->numberWindowDown != p0->numberWindowDown
            || p->numberWindowAcross != p0->numberWindowAcross
            || p->numberWindowDownInChunk != p0->numberWindowDownInChunk
            || p->numberWindowAcrossInChunk != p0->numberWindowAcrossInChunk)
            throw std::invalid_argument("All layers must have the same numbers of windows and windows in a chunk");
        for (const auto &name : {p->offsetImageName, p->grossOffsetImageName, p->snrImageName,
                                 p->covImageName, p->peakValueImageName})
            if (!outputs.insert(name).second)
                throw std::invalid_argument("Output file " + name + " is used more than once");
    }
}

} // namespace

/**
 *  Run ampcor
 */
void cuAmpcorController::runAmpcor()
{
    runAmpcorLayers({this});
}

/**
 * Run ampcor for several layers (parameter sets) sharing the images and the chunk partition
 * Each chunk is loaded once and processed by all layers. The device (deviceID) and the
 * number of workers (nStreams or nThreads) are taken from the first layer.
 */
void cuAmpcorController::runAmpcorLayers(const std::vector<cuAmpcorController *> &layers)
{
    std::vector<cuAmpcorParameter *> params;
    for (auto *layer : layers)
        params.push_back(layer->param);
    // check whether the parameters are set up and compatible
    checkLayers(params);
    cuAmpcorParameter *param = params.front();
    const int nLayers = params.size();

    // initialize the device (gpu id) or the cpu threads
    const int deviceID = backendInit(param);
    for (auto *p : params)
        p->deviceID = deviceID;

    // reference and secondary images
    int mmapSizeInGB = 0;
    for (const auto *p : params)
        mmapSizeInGB = std::max(mmapSizeInGB, p->mmapSizeInGB);
    // TODO: selecting band
    std::cout << "Opening reference image " << param->referenceImageName << "...\n";
    auto referenceImage = SlcImage::open(param->referenceImageName, param->referenceImageReader,
        param->referenceImageHeight, param->referenceImageWidth,
        param->referenceImageDataType*sizeof(float), mmapSizeInGB);
    std::cout << "Opening secondary image " << param->secondaryImageName << "...\n";
    auto secondaryImage = SlcImage::open(param->secondaryImageName, param->secondaryImageReader,
        param->secondaryImageHeight, param->secondaryImageWidth,
        param->secondaryImageDataType*sizeof(float), mmapSizeInGB);

    // the area of each chunk to load, covering the windows of all layers
    const auto footprints = cuAmpcorChunkLoader::footprints(
        std::vector<const cuAmpcorParameter *>(params.begin(), params.end()));

    // output images of each layer
    std::vector<std::unique_ptr<LayerImages>> images;
    for (const auto *p : params)
        images.push_back(std::make_unique<LayerImages>(p));

    // a worker (cuda stream, or cpu thread) with its own chunk loader and processors (one per layer)
    struct Worker {
        stream_t stream;
        std::unique_ptr<cuAmpcorChunkLoader> loader;
        std::vector<std::unique_ptr<cuAmpcorProcessor>> processors;
        Worker() : stream(backendCreateStream()) {}
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;
        ~Worker() {
            // cufft plans etc are stream dependent, need to be deleted before the stream is destroyed
            processors.clear();
            loader.reset();
            backendDestroyStream(stream);
        }
    };

    // set up the workers
    const int nWorkers = backendNumWorkers(param);
    std::vector<std::unique_ptr<Worker>> workers(nWorkers);
    for(auto &worker : workers)
    {
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
    stream_t stream = workers[0]->stream;

    const int nChunksDown = param->numberChunkDown;
    const int nChunksAcross = param->numberChunkAcross;
    const int nChunks = nChunksDown*nChunksAcross;

    // report info
    std::cout << "Total number of windows (azimuth x range):  "
        << param->numberWindowDown << " x " << param->numberWindowAcross
        << std::endl;
    std::cout << "to be processed in the number of chunks: "
        << nChunksDown << " x " << nChunksAcross  << std::endl;
    if (nLayers > 1)
        std::cout << "for " << nLayers << " layers sharing the loaded chunks" << std::endl;

    // iterate over all chunks
    // for GPU, chunks are assigned to cuda streams in turn (and the pragma is ignored);
    // for CPU, chunks are processed in parallel by openmp threads
    // exceptions can't propagate out of an openmp region; keep the first one and rethrow later
    const int message_interval = std::max(nChunksDown/10, 1);
    std::exception_ptr error;
    std::atomic<bool> failed(false);
    #pragma omp parallel for schedule(dynamic) num_threads(nWorkers)
    for(int k = 0; k < nChunks; k++)
    {
        if (failed) continue;
        const int i = k / nChunksAcross;
        const int j = k % nChunksAcross;
        if(j == 0 && i%message_interval == 0)
            printf("Processing chunks (%d, x) - (%d, x) out of %d\n",
                i+1, std::min(nChunksDown, i+message_interval), nChunksDown);
        try {
            auto &worker = workers[backendWorkerId(k, nWorkers)];
            // load the chunk once, and process it for all layers
            const cuAmpcorChunk &chunk = worker->loader->load(k);
            for (auto &processor : worker->processors)
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

    // wait all workers are done
    backendSynchronize();

    // save the outputs of each layer
    for (int l = 0; l < nLayers; l++)
        images[l]->write(params[l], stream);
}
// end of file


} // namespace