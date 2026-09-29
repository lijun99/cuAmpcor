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
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

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
    // initialize the device (gpu id) or the cpu threads
    param->deviceID = backendInit(param);

    // reference and secondary images
    // TODO: selecting band
    std::cout << "Opening reference image " << param->referenceImageName << "...\n";
    SlcImage *referenceImage = new SlcImage(param->referenceImageName, param->referenceImageHeight, param->referenceImageWidth,
        param->referenceImageDataType*sizeof(float), param->mmapSizeInGB);
    std::cout << "Opening secondary image " << param->secondaryImageName << "...\n";
    SlcImage *secondaryImage = new SlcImage(param->secondaryImageName, param->secondaryImageHeight, param->secondaryImageWidth,
        param->secondaryImageDataType*sizeof(float), param->mmapSizeInGB);

    cuArrays<real2_type> *offsetImage, *offsetImageRun;
    cuArrays<real_type> *snrImage, *snrImageRun;
    cuArrays<real3_type> *covImage, *covImageRun;
    cuArrays<real_type> *peakValueImage, *peakValueImageRun;

    // nWindowsDownRun is defined as numberChunk * numberWindowInChunk
    // It may be bigger than the actual number of windows
    int nWindowsDownRun = param->numberChunkDown * param->numberWindowDownInChunk;
    int nWindowsAcrossRun = param->numberChunkAcross * param->numberWindowAcrossInChunk;

    offsetImageRun = new cuArrays<real2_type>(nWindowsDownRun, nWindowsAcrossRun);
    offsetImageRun->allocate();

    snrImageRun = new cuArrays<real_type>(nWindowsDownRun, nWindowsAcrossRun);
    snrImageRun->allocate();

    covImageRun = new cuArrays<real3_type>(nWindowsDownRun, nWindowsAcrossRun);
    covImageRun->allocate();

    peakValueImageRun = new cuArrays<real_type>(nWindowsDownRun, nWindowsAcrossRun);
    peakValueImageRun->allocate();

    // Offset fields.
    offsetImage = new cuArrays<real2_type>(param->numberWindowDown, param->numberWindowAcross);
    offsetImage->allocate();

    // SNR.
    snrImage = new cuArrays<real_type>(param->numberWindowDown, param->numberWindowAcross);
    snrImage->allocate();

    // Variance.
    covImage = new cuArrays<real3_type>(param->numberWindowDown, param->numberWindowAcross);
    covImage->allocate();

    // Correlation surface peak value
    peakValueImage = new cuArrays<real_type>(param->numberWindowDown, param->numberWindowAcross);
    peakValueImage->allocate();



    // set up the workers (cuda streams, or cpu threads)
    const int nWorkers = backendNumWorkers(param);
    std::vector<stream_t> streams(nWorkers);
    std::vector<std::unique_ptr<cuAmpcorProcessor>> chunk(nWorkers);
    // iterate over workers
    for(int iworker=0; iworker < nWorkers; iworker++)
    {
        // create each stream
        streams[iworker] = backendCreateStream();
        // create the chunk processor for each worker
        chunk[iworker]= cuAmpcorProcessor::create(param->workflow, param, referenceImage, secondaryImage,
            offsetImageRun, snrImageRun, covImageRun, peakValueImageRun, streams[iworker]);
    }

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
    const int message_interval = std::max(nChunksDown/10, 1);
    #pragma omp parallel for schedule(dynamic) num_threads(nWorkers)
    for(int k = 0; k < nChunks; k++)
    {
        const int i = k / nChunksAcross;
        const int j = k % nChunksAcross;
        if(j == 0 && i%message_interval == 0)
            printf("Processing chunks (%d, x) - (%d, x) out of %d\n",
                i+1, std::min(nChunksDown, i+message_interval), nChunksDown);
        chunk[backendWorkerId(k, nWorkers)]->run(i, j);
    }

    // wait all workers are done
    backendSynchronize();

    // extraction of the run images to output images
    cuArraysCopyExtract(offsetImageRun, offsetImage, make_int2(0,0), streams[0]);
    cuArraysCopyExtract(snrImageRun, snrImage, make_int2(0,0), streams[0]);
    cuArraysCopyExtract(covImageRun, covImage, make_int2(0,0), streams[0]);
    cuArraysCopyExtract(peakValueImageRun, peakValueImage, make_int2(0,0), streams[0]);

    /* save the offsets and gross offsets */
    // copy the offset to host
    offsetImage->allocateHost();
    offsetImage->copyToHost(streams[0]);
    // construct the gross offset
    cuArrays<real2_type> *grossOffsetImage = new cuArrays<real2_type>(param->numberWindowDown, param->numberWindowAcross);
    grossOffsetImage->allocateHost();
    for(int i=0; i< param->numberWindows; i++)
        grossOffsetImage->hostData[i] = make_real2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);

    // check whether to merge gross offset
    if (param->mergeGrossOffset)
    {
        // if merge, add the gross offsets to offset
        for(int i=0; i< param->numberWindows; i++)
            offsetImage->hostData[i] += grossOffsetImage->hostData[i];
    }
    // output both offset and gross offset
    offsetImage->outputHostToFile(param->offsetImageName);
    grossOffsetImage->outputHostToFile(param->grossOffsetImageName);
    delete grossOffsetImage;

    // save the snr/cov images
    snrImage->outputToFile(param->snrImageName, streams[0]);
    covImage->outputToFile(param->covImageName, streams[0]);
    peakValueImage->outputToFile(param->peakValueImageName, streams[0]);

    // Delete arrays.
    delete offsetImage;
    delete snrImage;
    delete covImage;
    delete peakValueImage;

    delete offsetImageRun;
    delete snrImageRun;
    delete covImageRun;
    delete peakValueImageRun;

    for (int iworker=0; iworker < nWorkers; iworker++)
    {
        // cufftplan etc are stream dependent, need to be deleted before stream is destroyed
        chunk[iworker].reset();
        backendDestroyStream(streams[iworker]);
    }

    delete referenceImage;
    delete secondaryImage;

}
// end of file
