/**
 * @file cuAmpcorParameter.cu
 * Input parameters for ampcor
 */

#include "public.h"
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/**
 * To determine other process parameters after reading essential parameters from python
 */

void cuAmpcorParameter::setupParameters()
{
    // workflow specific parameter settings
    //
    switch (workflow) {
        case Workflow::TwoPass:
            _setupParameters_TwoPass();
            break;
        case Workflow::OnePass:
            _setupParameters_OnePass();
            break;
        default:
            throw std::invalid_argument("Unsupported workflow");
    }

    // common parameter settings
    numberWindows = numberWindowDown*numberWindowAcross;
    if(numberWindows <=0) {
        throw std::invalid_argument("Incorrect number of windows! ("
            + std::to_string(numberWindowDown) + ", " + std::to_string(numberWindowAcross) + ")");
    }

    // automatic (0 or negative) numbers of windows in a chunk: the backend default,
    // limited by the numbers of windows
    // (the inputs are kept, so that the setup can be repeated after changing the parameters)
    windowsDownInChunk = numberWindowDownInChunk;
    windowsAcrossInChunk = numberWindowAcrossInChunk;
    if(windowsDownInChunk <= 0 || windowsAcrossInChunk <= 0) {
        int down, across;
        backendDefaultChunkSize(this, down, across);
        if(windowsDownInChunk <= 0)
            windowsDownInChunk = std::min(down, numberWindowDown);
        if(windowsAcrossInChunk <= 0)
            windowsAcrossInChunk = std::min(across, numberWindowAcross);
    }

    numberChunkDown = divUp(numberWindowDown, windowsDownInChunk);
    numberChunkAcross = divUp(numberWindowAcross, windowsAcrossInChunk);
    numberChunks = numberChunkDown*numberChunkAcross;
    allocateArrays();
}

void cuAmpcorParameter::_setupParameters_TwoPass()
{
    // Size to extract the raw correlation surface for snr/cov
    corrRawZoomInHeight = std::min(corrStatWindowSize, 2*halfSearchRangeDownRaw+1);
    corrRawZoomInWidth = std::min(corrStatWindowSize, 2*halfSearchRangeAcrossRaw+1);

    // Size to extract the resampled correlation surface for oversampling
    // users should use 16 for zoomWindowSize, no need to multiply by 2
    // zoomWindowSize *= rawDataOversamplingFactor; //8 * 2
    // to check the search range
    int corrSurfaceActualSize =
        std::min(halfSearchRangeAcrossRaw, halfSearchRangeDownRaw)*
        2*rawDataOversamplingFactor;
    corrZoomWindowSize = std::min(zoomWindowSize, corrSurfaceActualSize);

    halfZoomWindowSizeRaw = corrZoomWindowSize/(2*rawDataOversamplingFactor); // 8*2/(2*2) = 4

    windowSizeWidth = windowSizeWidthRaw*rawDataOversamplingFactor;  //
    windowSizeHeight = windowSizeHeightRaw*rawDataOversamplingFactor;

    searchWindowSizeWidthRaw =  windowSizeWidthRaw + 2*halfSearchRangeAcrossRaw;
    searchWindowSizeHeightRaw = windowSizeHeightRaw + 2*halfSearchRangeDownRaw;

    searchWindowSizeWidthRawZoomIn = windowSizeWidthRaw + 2*halfZoomWindowSizeRaw;
    searchWindowSizeHeightRawZoomIn = windowSizeHeightRaw + 2*halfZoomWindowSizeRaw;

    searchWindowSizeWidth = searchWindowSizeWidthRawZoomIn*rawDataOversamplingFactor;
    searchWindowSizeHeight = searchWindowSizeHeightRawZoomIn*rawDataOversamplingFactor;

    windowSizeWidthRawEnlarged = windowSizeWidthRaw;
    windowSizeHeightRawEnlarged = windowSizeHeightRaw;

    // loading offsets
    referenceLoadingOffsetDown = 0;
    referenceLoadingOffsetAcross = 0;
    // secondary loading offset relative to reference
    secondaryLoadingOffsetDown = -halfSearchRangeDownRaw;
    secondaryLoadingOffsetAcross = -halfSearchRangeAcrossRaw;

}

void cuAmpcorParameter::_setupParameters_OnePass()
{

    // template window size (after antialiasing oversampling)
    windowSizeWidth = windowSizeWidthRaw*rawDataOversamplingFactor;  //
    windowSizeHeight = windowSizeHeightRaw*rawDataOversamplingFactor;

    // Size to extract the correlation surface (at the raw pixel spacing) for snr, as in the two-pass
    // workflow, within the search range (before adding the extra margin below)
    corrRawZoomInHeight = std::min(corrStatWindowSize, 2*halfSearchRangeDownRaw+1);
    corrRawZoomInWidth = std::min(corrStatWindowSize, 2*halfSearchRangeAcrossRaw+1);

    // serve as extra margin for search range
    corrZoomWindowSize = zoomWindowSize;
    halfZoomWindowSizeRaw = zoomWindowSize/(2*rawDataOversamplingFactor);

    // add extra search range to ensure enough area to oversample correlation surface
    // (kept apart from the input search range, so that the setup can be repeated)
    const int halfSearchRangeDownExtended = halfSearchRangeDownRaw + halfZoomWindowSizeRaw;
    const int halfSearchRangeAcrossExtended = halfSearchRangeAcrossRaw + halfZoomWindowSizeRaw;

    // search window size before antialiasing oversampling
    searchWindowSizeWidthRaw =  windowSizeWidthRaw + 2*halfSearchRangeAcrossExtended;
    searchWindowSizeHeightRaw = windowSizeHeightRaw + 2*halfSearchRangeDownExtended;

    windowSizeHeightRawEnlarged = searchWindowSizeHeightRaw;
    windowSizeWidthRawEnlarged = searchWindowSizeWidthRaw;

    // search window size (after antialiasing oversampling)
    searchWindowSizeWidth = searchWindowSizeWidthRaw*rawDataOversamplingFactor;
    searchWindowSizeHeight = searchWindowSizeHeightRaw*rawDataOversamplingFactor;

    windowSizeHeightEnlarged = searchWindowSizeHeight;
    windowSizeWidthEnlarged = searchWindowSizeWidth;

    // loading offsets
    // reference size matching the secondary size
    referenceLoadingOffsetDown = -halfSearchRangeDownExtended;
    referenceLoadingOffsetAcross = -halfSearchRangeAcrossExtended;
    // secondary loading offset relative to reference
    secondaryLoadingOffsetDown = 0;
    secondaryLoadingOffsetAcross = 0;

    // correlation surface size
    corrWindowSize = make_int2(searchWindowSizeHeight - windowSizeHeight + 1,
                               searchWindowSizeWidth - windowSizeWidth +1);
    // check zoom in window size, if larger, issue a warning
    if(zoomWindowSize >= corrWindowSize.x || zoomWindowSize >= corrWindowSize.y)
        logWarning("zoomWindowSize ", zoomWindowSize, " is bigger than the original correlation surface size (",
            corrWindowSize.x, ", ", corrWindowSize.y, ")");
    // use the smaller values
    corrZoomInSize = make_int2(std::min(zoomWindowSize+1, corrWindowSize.x),
                               std::min(zoomWindowSize+1, corrWindowSize.y));

    // oversampled correlation surface size
    corrZoomInOversampledSize = make_int2(corrZoomInSize.x * corrSurfaceOverSamplingFactor,
                                              corrZoomInSize.y * corrSurfaceOverSamplingFactor);

    int totalOS = rawDataOversamplingFactor*corrSurfaceOverSamplingFactor;
    // search start : image center - 1 pixel * totalOS
    corrZoomInOversampledSearchStart = make_int2((corrZoomInSize.x-1)*corrSurfaceOverSamplingFactor/2 - totalOS,
        (corrZoomInSize.y-1)*corrSurfaceOverSamplingFactor/2 - totalOS);
    // search range \pm 1 pixel * totalOS
    corrZoomInOversampledSearchRange = make_int2(2*totalOS, 2*totalOS);

    logDebug("zoomWindowSize is (", corrZoomInSize.x, ", ", corrZoomInSize.y, ")");

}

void cuAmpcorParameter::allocateArrays()
{
    // (re)size arrays, discarding values from previous setups, if any
    chunkStartPixelsSet = false;

    for (auto *array : {&grossOffsetDown, &grossOffsetAcross,
                        &referenceStartPixelDown, &referenceStartPixelAcross,
                        &secondaryStartPixelDown, &secondaryStartPixelAcross})
        array->assign(numberWindows, 0);

    for (auto *array : {&referenceChunkStartPixelDown, &referenceChunkStartPixelAcross,
                        &secondaryChunkStartPixelDown, &secondaryChunkStartPixelAcross,
                        &referenceChunkHeight, &referenceChunkWidth,
                        &secondaryChunkHeight, &secondaryChunkWidth})
        array->assign(numberChunks, 0);
}

/// check whether the arrays for windows/chunks are allocated (by setupParameters)
void cuAmpcorParameter::checkArraysAllocated() const
{
    if (grossOffsetDown.empty())
        throw std::logic_error("Parameters are not set up; call setupParams() first");
}

/// check whether the parameters are ready to run (setupParameters and setStartPixels are called)
void cuAmpcorParameter::checkReadyToRun() const
{
    checkArraysAllocated();
    if (!chunkStartPixelsSet)
        throw std::logic_error("Window starting pixels are not set; call setConstantGrossOffset() "
            "or setVaryingGrossOffset() after setupParams()");
}

// ****************
// make reference window the same as secondary for oversampling
// ****************

/// Set starting pixels for reference and secondary windows from arrays
/// set also gross offsets between reference and secondary windows
///
void cuAmpcorParameter::setStartPixels(const int *mStartD, const int *mStartA, const int *gOffsetD, const int *gOffsetA)
{
    checkArraysAllocated();
    for(int i=0; i<numberWindows; i++)
    {
        referenceStartPixelDown[i] = mStartD[i] + referenceLoadingOffsetDown;
        grossOffsetDown[i] = gOffsetD[i];
        secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] + secondaryLoadingOffsetDown;
        referenceStartPixelAcross[i] = mStartA[i] + referenceLoadingOffsetAcross;
        grossOffsetAcross[i] = gOffsetA[i];
        secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] + secondaryLoadingOffsetAcross;
    }
    setChunkStartPixels();
}

/// set starting pixels for each window with a varying gross offset
void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, const int *gOffsetD, const int *gOffsetA)
{
    checkArraysAllocated();
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            referenceStartPixelDown[i] = mStartD + row*skipSampleDownRaw + referenceLoadingOffsetDown;
            grossOffsetDown[i] = gOffsetD[i];
            secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] + secondaryLoadingOffsetDown;
            referenceStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw + referenceLoadingOffsetAcross;
            grossOffsetAcross[i] = gOffsetA[i];
            secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] + secondaryLoadingOffsetAcross;
        }
    }
    setChunkStartPixels();
}

/// set starting pixels for each window with a constant gross offset
void cuAmpcorParameter::setStartPixels(int mStartD, int mStartA, int gOffsetD, int gOffsetA)
{
    checkArraysAllocated();
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            referenceStartPixelDown[i] = mStartD + row*skipSampleDownRaw + referenceLoadingOffsetDown;
            grossOffsetDown[i] = gOffsetD;
            secondaryStartPixelDown[i] = referenceStartPixelDown[i] + grossOffsetDown[i] + secondaryLoadingOffsetDown;
            referenceStartPixelAcross[i] = mStartA + col*skipSampleAcrossRaw + referenceLoadingOffsetAcross;
            grossOffsetAcross[i] = gOffsetA;
            secondaryStartPixelAcross[i] = referenceStartPixelAcross[i] + grossOffsetAcross[i] + secondaryLoadingOffsetAcross;
        }
    }
    setChunkStartPixels();
}

/// set starting pixels for each chunk
void cuAmpcorParameter::setChunkStartPixels()
{
    checkArraysAllocated();

    maxReferenceChunkHeight = 0;
    maxReferenceChunkWidth = 0;
    maxSecondaryChunkHeight = 0;
    maxSecondaryChunkWidth = 0;

    for(int ichunk=0; ichunk <numberChunkDown; ichunk++)
    {
        for (int jchunk =0; jchunk<numberChunkAcross; jchunk++)
        {
            // index of chunk
            int idxChunk = ichunk*numberChunkAcross+jchunk;
            // variables to keep track of the first(s) and last(e) pixels in the chunk
            int mChunkSD = referenceImageHeight;
            int mChunkSA = referenceImageWidth;
            int mChunkED = 0;
            int mChunkEA = 0;
            int sChunkSD = secondaryImageHeight;
            int sChunkSA = secondaryImageWidth;
            int sChunkED = 0;
            int sChunkEA = 0;

            int numberWindowDownInChunkRun = windowsDownInChunk;
            int numberWindowAcrossInChunkRun = windowsAcrossInChunk;
            // modify the number of windows in last chunk
            if(ichunk == numberChunkDown -1)
                numberWindowDownInChunkRun = numberWindowDown - windowsDownInChunk*(numberChunkDown -1);
            if(jchunk == numberChunkAcross -1)
                numberWindowAcrossInChunkRun = numberWindowAcross - windowsAcrossInChunk*(numberChunkAcross -1);

            // iterate over windows in the chunk to find the starting pixels and the chunk height/windowSizeWidth
            // these parameters are needed to copy the chunk from the whole image file
            for(int i=0; i<numberWindowDownInChunkRun; i++)
            {
                for(int j=0; j<numberWindowAcrossInChunkRun; j++)
                {
                    int idxWindow = (ichunk*windowsDownInChunk+i)*numberWindowAcross + (jchunk*windowsAcrossInChunk+j);
                    int vpixel = referenceStartPixelDown[idxWindow];
                    if(mChunkSD > vpixel) mChunkSD = vpixel;
                    if(mChunkED < vpixel) mChunkED = vpixel;
                    vpixel = referenceStartPixelAcross[idxWindow];
                    if(mChunkSA > vpixel) mChunkSA = vpixel;
                    if(mChunkEA < vpixel) mChunkEA = vpixel;
                    vpixel = secondaryStartPixelDown[idxWindow];
                    if(sChunkSD > vpixel) sChunkSD = vpixel;
                    if(sChunkED < vpixel) sChunkED = vpixel;
                    vpixel = secondaryStartPixelAcross[idxWindow];
                    if(sChunkSA > vpixel) sChunkSA = vpixel;
                    if(sChunkEA < vpixel) sChunkEA = vpixel;
                }
            }

            // check whether the starting pixel exceeds the image range
            if (mChunkSD < 0) mChunkSD = 0;
            if (mChunkSA < 0) mChunkSA = 0;
            if (sChunkSD < 0) sChunkSD = 0;
            if (sChunkSA < 0) sChunkSA = 0;
            // set and check the last pixel
            mChunkED += windowSizeHeightRawEnlarged;
            if (mChunkED > referenceImageHeight) mChunkED = referenceImageHeight;
            mChunkEA += windowSizeWidthRawEnlarged;
            if (mChunkEA > referenceImageWidth) mChunkEA = referenceImageWidth;
            sChunkED += searchWindowSizeHeightRaw;
            if (sChunkED > secondaryImageHeight) sChunkED = secondaryImageHeight;
            sChunkEA += searchWindowSizeWidthRaw;
            if (sChunkEA > secondaryImageWidth) sChunkEA = secondaryImageWidth;
            // chunks entirely outside the image have zero size (and will be filled with zeros)
            if (mChunkED < mChunkSD) mChunkED = mChunkSD;
            if (mChunkEA < mChunkSA) mChunkEA = mChunkSA;
            if (sChunkED < sChunkSD) sChunkED = sChunkSD;
            if (sChunkEA < sChunkSA) sChunkEA = sChunkSA;
            // set the starting pixel and size of the chunk
            referenceChunkStartPixelDown[idxChunk]   = mChunkSD;
            referenceChunkStartPixelAcross[idxChunk] = mChunkSA;
            secondaryChunkStartPixelDown[idxChunk]    = sChunkSD;
            secondaryChunkStartPixelAcross[idxChunk]  = sChunkSA;
            referenceChunkHeight[idxChunk] = mChunkED - mChunkSD;
            referenceChunkWidth[idxChunk]  = mChunkEA - mChunkSA;
            secondaryChunkHeight[idxChunk]  = sChunkED - sChunkSD;
            secondaryChunkWidth[idxChunk]   = sChunkEA - sChunkSA;
            // search the max chunk size, used to determine the allocated read buffer size
            if(maxReferenceChunkHeight < referenceChunkHeight[idxChunk]) maxReferenceChunkHeight = referenceChunkHeight[idxChunk];
            if(maxReferenceChunkWidth  < referenceChunkWidth[idxChunk] ) maxReferenceChunkWidth  = referenceChunkWidth[idxChunk];
            if(maxSecondaryChunkHeight  < secondaryChunkHeight[idxChunk]) maxSecondaryChunkHeight = secondaryChunkHeight[idxChunk];
            if(maxSecondaryChunkWidth   < secondaryChunkWidth[idxChunk] ) maxSecondaryChunkWidth  = secondaryChunkWidth[idxChunk];
        }
    }
    chunkStartPixelsSet = true;
}

/// check whether reference and secondary windows are within the image range
// now issue warning rather than errors
void cuAmpcorParameter::checkPixelInImageRange() const
{
// check range is no longer required, but offered as an option in DEBUG
#ifdef CUAMPCOR_DEBUG
    int endPixel;
    for(int row=0; row<numberWindowDown; row++)
    {
        for(int col = 0; col < numberWindowAcross; col++)
        {
            int i = row*numberWindowAcross + col;
            if(referenceStartPixelDown[i] <0)
            {
                logWarning("Reference Window start pixel out of range in Down, window (", row, ",", col, "), pixel ", referenceStartPixelDown[i]);
            }
            if(referenceStartPixelAcross[i] <0)
            {
                logWarning("Reference Window start pixel out of range in Across, window (", row, ",", col, "), pixel ", referenceStartPixelAcross[i]);
            }
            endPixel = referenceStartPixelDown[i] + windowSizeHeightRaw;
            if(endPixel >= referenceImageHeight)
            {
                logWarning("Reference Window end pixel out of range in Down, window (", row, ",", col, "), pixel ", endPixel);
            }
            endPixel = referenceStartPixelAcross[i] + windowSizeWidthRaw;
            if(endPixel >= referenceImageWidth)
            {
                logWarning("Reference Window end pixel out of range in Across, window (", row, ",", col, "), pixel ", endPixel);
            }
            //secondary
            if(secondaryStartPixelDown[i] <0)
            {
                logWarning("Secondary Window start pixel out of range in Down, window (", row, ",", col, "), pixel ", secondaryStartPixelDown[i]);
            }
            if(secondaryStartPixelAcross[i] <0)
            {
                logWarning("Secondary Window start pixel out of range in Across, window (", row, ",", col, "), pixel ", secondaryStartPixelAcross[i]);
            }
            endPixel = secondaryStartPixelDown[i] + searchWindowSizeHeightRaw;
            if(endPixel >= secondaryImageHeight)
            {
                logWarning("Secondary Window end pixel out of range in Down, window (", row, ",", col, "), pixel ", endPixel);
            }
            endPixel = secondaryStartPixelAcross[i] + searchWindowSizeWidthRaw;
            if(endPixel >= secondaryImageWidth)
            {
                logWarning("Secondary Window end pixel out of range in Across, window (", row, ",", col, "), pixel ", endPixel);
            }

        }
    }
#endif // CUAMPCOR_DEBUG
}

// end of file

} // namespace