#include "cuAmpcorProcessor.h"
#include "cuAmpcorProcessorTwoPass.h"
#include "cuAmpcorProcessorOnePass.h"
#include "cuAmpcorUtil.h"

#include <stdexcept>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

// Factory method implementation
// create the batch processor for a given {workflow}
std::unique_ptr<cuAmpcorProcessor> cuAmpcorProcessor::create(int workflow,
    cuAmpcorParameter *param_,
    cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
    cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
    stream_t stream_)
{
    if (workflow == 0) {
        return std::unique_ptr<cuAmpcorProcessor>(new cuAmpcorProcessorTwoPass(
            param_, offsetImage_,
            snrImage_, covImage_, peakValueImage_, stream_));
    } else if (workflow == 1) {
        return std::unique_ptr<cuAmpcorProcessor>(new cuAmpcorProcessorOnePass(
            param_, offsetImage_,
            snrImage_, covImage_, peakValueImage_, stream_));
    } else {
        throw std::invalid_argument("Unsupported workflow");
    }
}

// constructor
cuAmpcorProcessor::cuAmpcorProcessor(cuAmpcorParameter *param_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        stream_t stream_)
    : param(param_),
    offsetImage(offsetImage_), snrImage(snrImage_), covImage(covImage_),
    peakValueImage(peakValueImage_), stream(stream_)
{
    ChunkOffsetDown = std::make_unique<cuArrays<int>>(param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetDown->allocate();
    ChunkOffsetDown->allocateHost();
    ChunkOffsetAcross = std::make_unique<cuArrays<int>>(param->numberWindowDownInChunk, param->numberWindowAcrossInChunk);
    ChunkOffsetAcross->allocate();
    ChunkOffsetAcross->allocateHost();
}

/// set chunk index
void cuAmpcorProcessor::setIndex(int idxDown_, int idxAcross_)
{
    idxChunkDown = idxDown_;
    idxChunkAcross = idxAcross_;
    idxChunk = idxChunkAcross + idxChunkDown*param->numberChunkAcross;

    if(idxChunkDown == param->numberChunkDown -1) {
        nWindowsDown = param->numberWindowDown - param->numberWindowDownInChunk*(param->numberChunkDown -1);
    }
    else {
        nWindowsDown = param->numberWindowDownInChunk;
    }

    if(idxChunkAcross == param->numberChunkAcross -1) {
        nWindowsAcross = param->numberWindowAcross - param->numberWindowAcrossInChunk*(param->numberChunkAcross -1);
    }
    else {
        nWindowsAcross = param->numberWindowAcrossInChunk;
    }
}

/// obtain the starting pixels for each chip
/// @param[in] oStartPixel start pixel locations for all chips
/// @param[out] rstartPixel  start pixel locations for chips within the chunk
void cuAmpcorProcessor::getRelativeOffset(int *rStartPixel, const std::vector<int> &oStartPixel, int diff)
{
    for(int i=0; i<param->numberWindowDownInChunk; ++i) {
        int iDown = i;
        if(i>=nWindowsDown) iDown = nWindowsDown-1;
        for(int j=0; j<param->numberWindowAcrossInChunk; ++j){
            int iAcross = j;
            if(j>=nWindowsAcross) iAcross = nWindowsAcross-1;
            int idxInChunk = iDown*param->numberWindowAcrossInChunk+iAcross;
            int idxInAll = (iDown+idxChunkDown*param->numberWindowDownInChunk)*param->numberWindowAcross
                + idxChunkAcross*param->numberWindowAcrossInChunk+iAcross;
            rStartPixel[idxInChunk] = oStartPixel[idxInAll] - diff;
        }
    }
}



/// copy the windows of the current chunk from a loaded chunk to a batch format (nImages, height, width)
/// @param[in] chunk the loaded chunk (of the reference or secondary image)
/// @param[in] startDown, startAcross starting pixels of all windows in the image
/// @param[out] batch the batch of windows
void cuAmpcorProcessor::copyToBatch(const cuAmpcorLoadedChunk &chunk,
    const std::vector<int> &startDown, const std::vector<int> &startAcross,
    cuArrays<complex_type> *batch)
{
    // check whether all pixels are outside the original image range
    if (chunk.empty()) {
        // yes, simply set the image to 0
        batch->setZero(stream);
        return;
    }

    // use cpu to compute the starting positions for each window relative to the chunk
    getRelativeOffset(ChunkOffsetDown->hostData, startDown, chunk.startDown);
    // copy the positions to gpu
    ChunkOffsetDown->copyToDevice(stream);
    // same for the across direction
    getRelativeOffset(ChunkOffsetAcross->hostData, startAcross, chunk.startAcross);
    ChunkOffsetAcross->copyToDevice(stream);

    // windows outside the chunk (image) are padded with zeros
    if (chunk.complexData) {
        // complex image (e.g., SLC)
        // if derampMethod = 0 (no deramp), take amplitudes; otherwise, copy complex data
        if (param->derampMethod == 0)
            cuArraysCopyToBatchAbsWithOffset(chunk.complexData, chunk.height, chunk.width,
                batch, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
        else
            cuArraysCopyToBatchWithOffset(chunk.complexData, chunk.height, chunk.width,
                batch, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    }
    else {
        // real image (e.g., TIFF), copied to complex
        cuArraysCopyToBatchWithOffsetR2C(chunk.realData, chunk.height, chunk.width,
            batch, ChunkOffsetDown->devData, ChunkOffsetAcross->devData, stream);
    }
}

} // namespace
