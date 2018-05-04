#include "cuAmpcorController.h"
#include <cstdio>
#include <iostream>

int main()
{
    cuAmpcorController *objOffset = new cuAmpcorController();
    cuAmpcorParameter *param = objOffset->param;
    
    param->algorithm = 0;
    param->deviceID = 0; 
    param->nStreams = 2; 
    param->masterImageName = "/home/ljzhu/share/slc_data/20131213.slc";
    param->masterImageHeight = 43008;
    param->masterImageWidth = 24320;
    param->slaveImageName = "/home/ljzhu/share/slc_data/20131221.slc";
    param->slaveImageHeight = 43008;
    param->slaveImageWidth = 24320;
    
    param->windowSizeWidthRaw = 64;
    param->windowSizeHeightRaw = 64;
    
    param->numberWindowDown = 1281;
    param->numberWindowAcross = 15;
    param->skipSampleDownRaw = 32;
    param->skipSampleAcrossRaw = 16;
    param->numberWindowDownInChunk = 20;
    param->numberWindowAcrossInChunk = 10;
    
    param->zoomWindowSize = 8;
    param->oversamplingMethod = 0;
    param->halfSearchRangeDownRaw = 20;
    param->halfSearchRangeAcrossRaw = 20;
    param->derampMethod = 0;
    
    param->mmapSizeInGB = 4;

    param->setupParameters();
    param->setStartPixels(1000, 1000, 642, -30);
    param->checkPixelInImageRange();
    std::cout << "here ok\n";
    objOffset->runAmpcor();
    
}

