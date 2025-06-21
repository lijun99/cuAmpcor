/**
 * @file cuAmpcorController.cu
 * @brief Implementations of cuAmpcorController
 */

// my declaration
#include "cuAmpcorController.h"

// dependencies
#include "SlcImage.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cuAmpcorProcessor.h"
#include "cuAmpcorUtil.h"
#include <cuda_runtime.h>
#include <iostream>

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
    // set the gpu id
    param->deviceID = gpuDeviceInit(param->deviceID);

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



    // set up the cuda streams
    cudaStream_t streams[param->nStreams];
    std::unique_ptr<cuAmpcorProcessor> chunk[param->nStreams];
    // iterate over cuda streams
    for(int istream=0; istream < param->nStreams; istream++)
    {
        // create each stream
        checkCudaErrors(cudaStreamCreate(&streams[istream]));
        // create the chunk processor for each stream
        chunk[istream]= cuAmpcorProcessor::create(param->workflow, param, referenceImage, secondaryImage,
            offsetImageRun, snrImageRun, covImageRun, peakValueImageRun, streams[istream]);
    }

    int nChunksDown = param->numberChunkDown;
    int nChunksAcross = param->numberChunkAcross;

    // report info
    std::cout << "Total number of windows (azimuth x range):  "
        << param->numberWindowDown << " x " << param->numberWindowAcross
        << std::endl;
    std::cout << "to be processed in the number of chunks: "
        << nChunksDown << " x " << nChunksAcross  << std::endl;

    // iterative over chunks down
    int message_interval = std::max(nChunksDown/10, 1);
    for(int i = 0; i<nChunksDown; i++)
    {
        if(i%message_interval == 0)
            std::cout << "Processing chunks (" << i+1 <<", x) - (" << std::min(nChunksDown, i+message_interval )
                << ", x) out of " << nChunksDown << std::endl;
        // iterate over chunks across
        for(int j=0; j<nChunksAcross; j+=param->nStreams)
        {
            // iterate over cuda streams to process chunks
            for(int istream = 0; istream < param->nStreams; istream++)
            {
                int chunkIdxAcross = j+istream;
                if(chunkIdxAcross < nChunksAcross) {
                    chunk[istream]->run(i, chunkIdxAcross);
                }
            }
        }
    }

    // wait all streams are done
    cudaDeviceSynchronize();

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

    for (int istream=0; istream < param->nStreams; istream++)
    {
        // cufftplan etc are stream dependent, need to be deleted before stream is destroyed
        chunk[istream].release();
        checkCudaErrors(cudaStreamDestroy(streams[istream]));
    }

    delete referenceImage;
    delete secondaryImage;

}
// end of file
