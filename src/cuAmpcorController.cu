// Implementation of cuAmpcorController

#include "cuAmpcorController.h"
#include "GDALImage.h"
#include "cuArrays.h"
#include "cudaUtil.h"
#include "cuAmpcorChunk.h"
#include "cuAmpcorUtil.h"
#include <iostream>

cuAmpcorController::cuAmpcorController() { param = new cuAmpcorParameter();}
cuAmpcorController::~cuAmpcorController() { delete param; }

void cuAmpcorController::runAmpcor() {

    // set the gpu id
    param->deviceID = gpuDeviceInit(param->deviceID);
    // initialize the gdal driver
    GDALAllRegister();
    // reference and secondary images; use band=1 as default
    // TODO: selecting band
    GDALImage *referenceImage = new GDALImage(param->referenceImageName, 1, param->mmapSizeInGB);
    GDALImage *secondaryImage = new GDALImage(param->secondaryImageName, 1, param->mmapSizeInGB);

    cuArrays<float2> *offsetImage, *offsetImageRun;
    cuArrays<float> *snrImage, *snrImageRun;
    cuArrays<float3> *covImage, *covImageRun;

    // For debugging.
    // cuArrays<int> *corrValidCountImage;
    // cuArrays<float> *corrSumImage;

    int nWindowsDownRun = param->numberChunkDown * param->numberWindowDownInChunk;
    int nWindowsAcrossRun = param->numberChunkAcross * param->numberWindowAcrossInChunk;

    //std::cout << "The number of windows to be processed (might be bigger) " << nWindowsDownRun << " x " << param->numberWindowDown << "\n";

    offsetImageRun = new cuArrays<float2>(nWindowsDownRun, nWindowsAcrossRun);
    offsetImageRun->allocate();

    snrImageRun = new cuArrays<float>(nWindowsDownRun, nWindowsAcrossRun);
    snrImageRun->allocate();

    covImageRun = new cuArrays<float3>(nWindowsDownRun, nWindowsAcrossRun);
    covImageRun->allocate();

    // Offsetfields.
    offsetImage = new cuArrays<float2>(param->numberWindowDown, param->numberWindowAcross);
    offsetImage->allocate();

    // SNR.
    snrImage = new cuArrays<float>(param->numberWindowDown, param->numberWindowAcross);
    snrImage->allocate();

    // Variance.
    covImage = new cuArrays<float3>(param->numberWindowDown, param->numberWindowAcross);
    covImage->allocate();

    cudaStream_t streams[param->nStreams];
    cuAmpcorChunk *chunk[param->nStreams];
    for(int ist=0; ist<param->nStreams; ist++)
    {
        cudaStreamCreate(&streams[ist]);
        chunk[ist]= new cuAmpcorChunk(param, referenceImage, secondaryImage, offsetImageRun, snrImageRun, covImageRun,
            streams[ist]);

    }

    int nChunksDown = param->numberChunkDown;
    int nChunksAcross = param->numberChunkAcross;

    std::cout << "Total number of windows (azimuth x range):  " <<param->numberWindowDown << " x " << param->numberWindowAcross  << std::endl;
    std::cout << "to be processed in the number of chunks: " <<nChunksDown << " x " << nChunksAcross  << std::endl;

    for(int i = 0; i<nChunksDown; i++)
    {
        std::cout << "Processing chunk (" << i <<", x" << ")" << std::endl;
        for(int j=0; j<nChunksAcross; j+=param->nStreams)
        {
			//std::cout << "Processing chunk(" << i <<", " << j <<")" << std::endl;
            for(int ist = 0; ist<param->nStreams; ist++)
            {
                if(j+ist < nChunksAcross) {

                    chunk[ist]->run(i, j+ist);
                }
            }
        }
    }

    cudaDeviceSynchronize();

    // Do extraction.
    cuArraysCopyExtract(offsetImageRun, offsetImage, make_int2(0,0), streams[0]);
    cuArraysCopyExtract(snrImageRun, snrImage, make_int2(0,0), streams[0]);
    cuArraysCopyExtract(covImageRun, covImage, make_int2(0,0), streams[0]);

    offsetImage->outputToFile(param->offsetImageName, streams[0]);
    snrImage->outputToFile(param->snrImageName, streams[0]);
    covImage->outputToFile(param->covImageName, streams[0]);


    outputGrossOffsets();

    // Delete arrays.
    delete offsetImage;
    delete snrImage;
    delete covImage;

    delete offsetImageRun;
    delete snrImageRun;
    delete covImageRun;

    for (int ist=0; ist<param->nStreams; ist++)
        delete chunk[ist];

    delete referenceImage;
    delete secondaryImage;

}

void cuAmpcorController::outputGrossOffsets()
{
    cuArrays<float2> *grossOffsets = new cuArrays<float2>(param->numberWindowDown, param->numberWindowAcross);
    grossOffsets->allocateHost();

    for(int i=0; i< param->numberWindows; i++)
        grossOffsets->hostData[i] = make_float2(param->grossOffsetDown[i], param->grossOffsetAcross[i]);
    grossOffsets->outputHostToFile(param->grossOffsetImageName);
    delete grossOffsets;
}


/*
void cuAmpcorController::setAlgorithm(int n) { param->algorithm = n; } // 0 - freq domain; 1 - time domain
int cuAmpcorController::getAlgorithm() { return param->algorithm; }
void cuAmpcorController::setDeviceID(int n) { param->deviceID = n; }
int cuAmpcorController::getDeviceID() { return param->deviceID; }
void cuAmpcorController::setNStreams(int n) { param->nStreams = n; }
int cuAmpcorController::getNStreams() { return param->nStreams; }
void cuAmpcorController::setWindowSizeHeight(int n) { param->windowSizeHeight = n; }
int cuAmpcorController::getWindowSizeHeight() { return param->windowSizeHeight; }
void cuAmpcorController::setWindowSizeWidth(int n) { param->windowSizeWidth = n; }
int cuAmpcorController::getWindowSizeWidth() { return param->windowSizeWidth; }
void cuAmpcorController::setSearchWindowSizeHeight(int n) { param->searchWindowSizeHeight = n; }
int cuAmpcorController::getSearchWindowSizeHeight() { return param->windowSizeHeight; }
void cuAmpcorController::setSearchWindowSizeWidth(int n) { param->searchWindowSizeWidth = n; }
void cuAmpcorController::setRawOversamplingFactor(int n) { param->rawDataOversamplingFactor = n; }
void cuAmpcorController::setZoomWindowSize(int n) { param->zoomWindowSize = n; }
void cuAmpcorController::setOversamplingFactor(int n) { param->oversamplingFactor = n; }
//void cuAmpcorController::setAcrossLooks(int n) { param->acrossLooks = n; } // 1 - single look
//void cuAmpcorController::setDownLooks(int n) { param->downLooks = n; } // 1 - single look
//void cuAmpcorController::setSkipSampleAcross(int n) { param->skipSampleAcross = n; }
//void cuAmpcorController::setSkipSampleDown(int n) { param->skipSampleDown = n; }
void cuAmpcorController::setSkipSampleAcrossRaw(int n) { param->skipSampleAcrossRaw = n; }
int cuAmpcorController::getSkipSampleAcrossRaw() { return param->skipSampleAcrossRaw; }
void cuAmpcorController::setSkipSampleDownRaw(int n) { param->skipSampleDownRaw = n; }
int cuAmpcorController::getSkipSampleDownRaw() { return param->skipSampleDownRaw; }
void cuAmpcorController::setNumberWindowDown(int n) { param->numberWindowDown = n; }
void cuAmpcorController::setNumberWindowAcross(int n) { param->numberWindowAcross = n; }
void cuAmpcorController::setNumberWindowDownInChunk(int n) { param->numberWindowDownInChunk = n; }
void cuAmpcorController::setNumberWindowAcrossInChunk(int n) { param->numberWindowAcrossInChunk = n; }
//void cuAmpcorController::setRangeSpacing1(float n) { param->rangeSpacing1 = n; } // deprecated
//void cuAmpcorController::setRangeSpacing2(float n) { param->rangeSpacing2 = n; } // deprecated
//void cuAmpcorController::setImageDatatype1(int n) { param->imageDataType1 = n; }
//void cuAmpcorController::setImageDatatype2(int n) { param->imageDataType2 = n; }
void cuAmpcorController::setThresholdSNR(float n) { param->thresholdSNR = n; } // deprecated(?)
//void cuAmpcorController::setThresholdCov(float n) { param->thresholdCov = n; } // deprecated(?)
//void cuAmpcorController::setBand1(int n) { param->band1 = n; }
//void cuAmpcorController::setBand2(int n) { param->band2 = n; }
void cuAmpcorController::setReferenceImageName(std::string s) { param->referenceImageName = s; }
std::string cuAmpcorController::getReferenceImageName() {return param->referenceImageName;}
void cuAmpcorController::setSecondaryImageName(std::string s) { param->secondaryImageName = s; }
std::string cuAmpcorController::getSecondaryImageName() {return param->secondaryImageName;}
void cuAmpcorController::setReferenceImageWidth(int n) { param->referenceImageWidth = n; }
void cuAmpcorController::setReferenceImageHeight(int n) { param->referenceImageHeight = n; }
void cuAmpcorController::setSecondaryImageWidth(int n) { param->secondaryImageWidth = n; }
void cuAmpcorController::setSecondaryImageHeight(int n) { param->secondaryImageHeight = n; }
//void cuAmpcorController::setReferenceStartPixelAcross(int n) { param->referenceStartPixelAcross = n; }
//void cuAmpcorController::setReferenceStartPixelDown(int n) { param->referenceStartPixelDown = n; }
//void cuAmpcorController::setSecondaryStartPixelAcross(int n) { param->secondaryStartPixelAcross = n; }
//void cuAmpcorController::setSecondaryStartPixelDown(int n) { param->secondaryStartPixelDown = n; }
//void cuAmpcorController::setGrossOffsetMethod(int n) { param->grossOffsetMethod = n; }
//int cuAmpcorController::getGrossOffsetMethod() { return param->grossOffsetMethod; }
//void cuAmpcorController::setAcrossGrossOffset(int n) { param->acrossGrossOffset = n; }
//void cuAmpcorController::setDownGrossOffset(int n) { param->downGrossOffset = n; }
//int* cuAmpcorController::getGrossOffsets() {return param->grossOffsets;}

void cuAmpcorController::setGrossOffsets(int *in, int size) {
    assert(size = 2*param->numberWindowAcross*param->numberWindowDown);
    if (param->grossOffsets == NULL)
        param->grossOffsets = (int *)malloc(size*sizeof(int));
    mempcpy(param->grossOffsets, in, size*sizeof(int));
    fprintf(stderr, "copy grossOffsets %d\n", size);
}
void cuAmpcorController::setOffsetImageName(std::string s) { param->offsetImageName = s; }
void cuAmpcorController::setSNRImageName(std::string s) { param->snrImageName = s; }
//void cuAmpcorController::setMargin(int n) { param->margin = n; }
void cuAmpcorController::setDerampMethod(int n) { param->derampMethod = n; }
int cuAmpcorController::getDerampMethod() { return param->derampMethod; }
*/
