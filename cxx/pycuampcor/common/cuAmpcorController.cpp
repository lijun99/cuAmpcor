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
#include <algorithm>
#include <atomic>
#include <exception>
#include <type_traits>
#include <cstdio>
#include <iostream>
#include <memory>
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
/**
 *  Run ampcor
 *
 *
 */
void cuAmpcorController::runAmpcor()
{
    // check whether the parameters are set up
    param->checkReadyToRun();

    // initialize the device (gpu id) or the cpu threads
    param->deviceID = backendInit(param);

    // reference and secondary images
    // TODO: selecting band
    std::cout << "Opening reference image " << param->referenceImageName << "...\n";
    auto referenceImage = std::make_unique<SlcImage>(param->referenceImageName,
        param->referenceImageHeight, param->referenceImageWidth,
        param->referenceImageDataType*sizeof(float), param->mmapSizeInGB);
    std::cout << "Opening secondary image " << param->secondaryImageName << "...\n";
    auto secondaryImage = std::make_unique<SlcImage>(param->secondaryImageName,
        param->secondaryImageHeight, param->secondaryImageWidth,
        param->secondaryImageDataType*sizeof(float), param->mmapSizeInGB);

    // allocate an array in (device) memory
    auto newArray = [](auto &ptr, int height, int width) {
        using T = typename std::remove_reference_t<decltype(ptr)>::element_type;
        ptr = std::make_unique<T>(height, width);
        ptr->allocate();
    };

    // nWindowsDownRun is defined as numberChunk * numberWindowInChunk
    // It may be bigger than the actual number of windows
    int nWindowsDownRun = param->numberChunkDown * param->numberWindowDownInChunk;
    int nWindowsAcrossRun = param->numberChunkAcross * param->numberWindowAcrossInChunk;

    std::unique_ptr<cuArrays<real2_type>> offsetImageRun;
    std::unique_ptr<cuArrays<real_type>> snrImageRun;
    std::unique_ptr<cuArrays<real3_type>> covImageRun;
    std::unique_ptr<cuArrays<real_type>> peakValueImageRun;
    newArray(offsetImageRun, nWindowsDownRun, nWindowsAcrossRun);
    newArray(snrImageRun, nWindowsDownRun, nWindowsAcrossRun);
    newArray(covImageRun, nWindowsDownRun, nWindowsAcrossRun);
    newArray(peakValueImageRun, nWindowsDownRun, nWindowsAcrossRun);

    // output images: offset fields, SNR, variance, and correlation surface peak value
    std::unique_ptr<cuArrays<real2_type>> offsetImage;
    std::unique_ptr<cuArrays<real_type>> snrImage;
    std::unique_ptr<cuArrays<real3_type>> covImage;
    std::unique_ptr<cuArrays<real_type>> peakValueImage;
    newArray(offsetImage, param->numberWindowDown, param->numberWindowAcross);
    newArray(snrImage, param->numberWindowDown, param->numberWindowAcross);
    newArray(covImage, param->numberWindowDown, param->numberWindowAcross);
    newArray(peakValueImage, param->numberWindowDown, param->numberWindowAcross);

    // a worker (cuda stream, or cpu thread) with its own chunk processor
    struct Worker {
        stream_t stream;
        std::unique_ptr<cuAmpcorProcessor> processor;
        Worker() : stream(backendCreateStream()) {}
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;
        ~Worker() {
            // cufft plans etc are stream dependent, need to be deleted before the stream is destroyed
            processor.reset();
            backendDestroyStream(stream);
        }
    };

    // set up the workers
    const int nWorkers = backendNumWorkers(param);
    std::vector<std::unique_ptr<Worker>> workers(nWorkers);
    for(auto &worker : workers)
    {
        worker = std::make_unique<Worker>();
        worker->processor = cuAmpcorProcessor::create(param->workflow, param, referenceImage.get(), secondaryImage.get(),
            offsetImageRun.get(), snrImageRun.get(), covImageRun.get(), peakValueImageRun.get(), worker->stream);
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
            workers[backendWorkerId(k, nWorkers)]->processor->run(i, j);
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

    // extraction of the run images to output images
    cuArraysCopyExtract(offsetImageRun.get(), offsetImage.get(), make_int2(0,0), stream);
    cuArraysCopyExtract(snrImageRun.get(), snrImage.get(), make_int2(0,0), stream);
    cuArraysCopyExtract(covImageRun.get(), covImage.get(), make_int2(0,0), stream);
    cuArraysCopyExtract(peakValueImageRun.get(), peakValueImage.get(), make_int2(0,0), stream);

    /* save the offsets and gross offsets */
    // copy the offset to host
    offsetImage->allocateHost();
    offsetImage->copyToHost(stream);
    // construct the gross offset
    cuArrays<real2_type> grossOffsetImage(param->numberWindowDown, param->numberWindowAcross);
    grossOffsetImage.allocateHost();
    for(int i=0; i< param->numberWindows; i++)
        grossOffsetImage.hostData[i] = make_real2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);

    // check whether to merge gross offset
    if (param->mergeGrossOffset)
    {
        // if merge, add the gross offsets to offset
        for(int i=0; i< param->numberWindows; i++)
            offsetImage->hostData[i] += grossOffsetImage.hostData[i];
    }
    // output both offset and gross offset
    offsetImage->outputHostToFile(param->offsetImageName);
    grossOffsetImage.outputHostToFile(param->grossOffsetImageName);

    // save the snr/cov images
    snrImage->outputToFile(param->snrImageName, stream);
    covImage->outputToFile(param->covImageName, stream);
    peakValueImage->outputToFile(param->peakValueImageName, stream);
}
// end of file


} // namespace