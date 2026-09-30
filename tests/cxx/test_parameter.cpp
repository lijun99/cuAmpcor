/**
 * @file test_parameter.cpp
 * @brief Tests for the parameters: derived sizes, window/chunk starting pixels and setup checks
 */

#include "test_util.h"
#include "cuAmpcorParameter.h"

#include <stdexcept>

namespace pycuampcor::PYCUAMPCOR_BACKEND::test {

static void setWindows(cuAmpcorParameter &p)
{
    p.windowSizeWidthRaw = 64;
    p.windowSizeHeightRaw = 32;
    p.halfSearchRangeDownRaw = 20;
    p.halfSearchRangeAcrossRaw = 10;
    p.rawDataOversamplingFactor = 2;
    p.zoomWindowSize = 16;
    p.corrSurfaceOverSamplingFactor = 16;
    p.numberWindowDown = 3;
    p.numberWindowAcross = 4;
}

TEST(ParameterTest, TwoPassSizes)
{
    cuAmpcorParameter p;
    setWindows(p);
    p.workflow = 0;
    p.setupParameters();
    EXPECT_EQ(p.zoomWindowSize, 16);
    EXPECT_EQ(p.halfZoomWindowSizeRaw, 4);
    EXPECT_EQ(p.windowSizeWidth, 128);
    EXPECT_EQ(p.windowSizeHeight, 64);
    EXPECT_EQ(p.searchWindowSizeWidthRaw, 64 + 2*10);
    EXPECT_EQ(p.searchWindowSizeHeightRaw, 32 + 2*20);
    EXPECT_EQ(p.searchWindowSizeWidthRawZoomIn, 64 + 2*4);
    EXPECT_EQ(p.searchWindowSizeHeightRawZoomIn, 32 + 2*4);
    EXPECT_EQ(p.searchWindowSizeWidth, (64 + 2*4)*2);
    EXPECT_EQ(p.searchWindowSizeHeight, (32 + 2*4)*2);
    EXPECT_EQ(p.secondaryLoadingOffsetDown, -20);
    EXPECT_EQ(p.secondaryLoadingOffsetAcross, -10);
    EXPECT_EQ(p.numberWindows, 12);
}

TEST(ParameterTest, TwoPassZoomWindowLimitedBySearchRange)
{
    cuAmpcorParameter p;
    setWindows(p);
    p.halfSearchRangeAcrossRaw = 3;
    p.workflow = 0;
    p.setupParameters();
    // min(half search range)*2*oversampling
    EXPECT_EQ(p.zoomWindowSize, 3*2*2);
}

TEST(ParameterTest, OnePassSizes)
{
    cuAmpcorParameter p;
    setWindows(p);
    p.workflow = 1;
    p.setupParameters();
    const int halfZoom = 16/(2*2);
    EXPECT_EQ(p.halfZoomWindowSizeRaw, halfZoom);
    // the search range is extended by the zoom-in window
    EXPECT_EQ(p.halfSearchRangeDownRaw, 20 + halfZoom);
    EXPECT_EQ(p.halfSearchRangeAcrossRaw, 10 + halfZoom);
    const int sw = 64 + 2*(10 + halfZoom), sh = 32 + 2*(20 + halfZoom);
    EXPECT_EQ(p.searchWindowSizeWidthRaw, sw);
    EXPECT_EQ(p.searchWindowSizeHeightRaw, sh);
    EXPECT_EQ(p.windowSizeWidthRawEnlarged, sw);
    EXPECT_EQ(p.windowSizeHeightRawEnlarged, sh);
    EXPECT_EQ(p.searchWindowSizeWidth, 2*sw);
    EXPECT_EQ(p.searchWindowSizeHeight, 2*sh);
    EXPECT_EQ(p.corrWindowSize.x, 2*sh - 64 + 1);
    EXPECT_EQ(p.corrWindowSize.y, 2*sw - 128 + 1);
    EXPECT_EQ(p.corrZoomInSize.x, 17);
    EXPECT_EQ(p.corrZoomInSize.y, 17);
    EXPECT_EQ(p.corrZoomInOversampledSize.x, 17*16);
    // search +/- 1 raw pixel around the center of the oversampled zoom-in surface
    const int totalOS = 2*16;
    EXPECT_EQ(p.corrZoomInOversampledSearchStart.x, 16*16/2 - totalOS);
    EXPECT_EQ(p.corrZoomInOversampledSearchRange.x, 2*totalOS);
    EXPECT_EQ(p.referenceLoadingOffsetDown, -(20 + halfZoom));
    EXPECT_EQ(p.referenceLoadingOffsetAcross, -(10 + halfZoom));
}

TEST(ParameterTest, SetupChecks)
{
    cuAmpcorParameter p;
    setWindows(p);
    // not set up yet
    EXPECT_THROW(p.checkReadyToRun(), std::logic_error);
    EXPECT_THROW(p.setStartPixels(0, 0, 0, 0), std::logic_error);
    p.setupParameters();
    // starting pixels not set
    EXPECT_THROW(p.checkReadyToRun(), std::logic_error);
    p.setStartPixels(30, 20, 0, 0);
    EXPECT_NO_THROW(p.checkReadyToRun());
    // setting up again requires the starting pixels again
    p.setupParameters();
    EXPECT_THROW(p.checkReadyToRun(), std::logic_error);

    cuAmpcorParameter bad;
    setWindows(bad);
    bad.workflow = 2;
    EXPECT_THROW(bad.setupParameters(), std::invalid_argument);
    cuAmpcorParameter empty;
    setWindows(empty);
    empty.numberWindowDown = 0;
    EXPECT_THROW(empty.setupParameters(), std::invalid_argument);
}

// the starting pixels of windows and chunks (to load from the images)
TEST(ParameterTest, WindowAndChunkPixels)
{
    cuAmpcorParameter p;
    p.workflow = 0;
    p.windowSizeWidthRaw = p.windowSizeHeightRaw = 16;
    p.halfSearchRangeDownRaw = p.halfSearchRangeAcrossRaw = 4;
    p.skipSampleDownRaw = p.skipSampleAcrossRaw = 16;
    p.zoomWindowSize = 8;
    p.referenceImageHeight = p.secondaryImageHeight = 100;
    p.referenceImageWidth = p.secondaryImageWidth = 120;
    // windows extending beyond the right edge, and a chunk entirely outside
    p.numberWindowDown = 4;
    p.numberWindowAcross = 8;
    p.numberWindowDownInChunk = 2;
    p.numberWindowAcrossInChunk = 2;
    p.setupParameters();
    const int startD = 10, startA = 50, grossD = 2, grossA = -3;
    p.setStartPixels(startD, startA, grossD, grossA);

    EXPECT_EQ(p.numberChunkDown, 2);
    EXPECT_EQ(p.numberChunkAcross, 4);
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 8; col++) {
            int i = row*8 + col;
            EXPECT_EQ(p.referenceStartPixelDown[i], startD + row*16);
            EXPECT_EQ(p.referenceStartPixelAcross[i], startA + col*16);
            EXPECT_EQ(p.secondaryStartPixelDown[i], startD + row*16 + grossD - 4);
            EXPECT_EQ(p.secondaryStartPixelAcross[i], startA + col*16 + grossA - 4);
            EXPECT_EQ(p.grossOffsetDown[i], grossD);
            EXPECT_EQ(p.grossOffsetAcross[i], grossA);
        }

    // the first chunk: windows (0-1, 0-1)
    EXPECT_EQ(p.referenceChunkStartPixelDown[0], 10);
    EXPECT_EQ(p.referenceChunkStartPixelAcross[0], 50);
    EXPECT_EQ(p.referenceChunkHeight[0], 16 + 16);
    EXPECT_EQ(p.referenceChunkWidth[0], 16 + 16);
    EXPECT_EQ(p.secondaryChunkStartPixelDown[0], 10 + 2 - 4);
    EXPECT_EQ(p.secondaryChunkStartPixelAcross[0], 50 - 3 - 4);
    EXPECT_EQ(p.secondaryChunkHeight[0], 16 + 24);
    EXPECT_EQ(p.secondaryChunkWidth[0], 16 + 24);

    // chunks are within the image ranges
    for (int c = 0; c < p.numberChunks; c++) {
        EXPECT_GE(p.referenceChunkStartPixelDown[c], 0);
        EXPECT_GE(p.referenceChunkStartPixelAcross[c], 0);
        EXPECT_GE(p.referenceChunkHeight[c], 0);
        EXPECT_GE(p.referenceChunkWidth[c], 0);
        EXPECT_GE(p.secondaryChunkHeight[c], 0);
        EXPECT_GE(p.secondaryChunkWidth[c], 0);
        // non-empty chunks (empty ones are filled with zeros)
        if (p.referenceChunkHeight[c] > 0 && p.referenceChunkWidth[c] > 0) {
            EXPECT_LE(p.referenceChunkStartPixelDown[c] + p.referenceChunkHeight[c], 100) << c;
            EXPECT_LE(p.referenceChunkStartPixelAcross[c] + p.referenceChunkWidth[c], 120) << c;
        }
        if (p.secondaryChunkHeight[c] > 0 && p.secondaryChunkWidth[c] > 0) {
            EXPECT_LE(p.secondaryChunkStartPixelDown[c] + p.secondaryChunkHeight[c], 100) << c;
            EXPECT_LE(p.secondaryChunkStartPixelAcross[c] + p.secondaryChunkWidth[c], 120) << c;
        }
        EXPECT_LE(p.referenceChunkHeight[c], p.maxReferenceChunkHeight);
        EXPECT_LE(p.referenceChunkWidth[c], p.maxReferenceChunkWidth);
    }
    // the third chunk across: windows 4-5 start at 114, 130; clamped at the right edge
    EXPECT_EQ(p.referenceChunkStartPixelAcross[2], 114);
    EXPECT_EQ(p.referenceChunkWidth[2], 120 - 114);
    // the last chunk across: windows 6-7 start at 146, 162, entirely outside
    EXPECT_EQ(p.referenceChunkWidth[3], 0);
    EXPECT_EQ(p.secondaryChunkWidth[3], 0);

    // copies own their arrays
    cuAmpcorParameter q = p;
    q.setStartPixels(0, 0, 0, 0);
    EXPECT_EQ(p.referenceStartPixelAcross[1], startA + 16);
    EXPECT_EQ(q.referenceStartPixelAcross[1], 16);

    // setting up again resizes the arrays to the new number of windows/chunks
    p.numberWindowAcross = 3;
    p.setupParameters();
    EXPECT_EQ(p.referenceStartPixelDown.size(), 4u*3u);
    EXPECT_EQ(p.referenceChunkHeight.size(), 2u*2u);
}

} // namespace
