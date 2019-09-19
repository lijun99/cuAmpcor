// -*- c++ -*-
///\brief

#ifndef __GDALIMAGE_H
#define __GDALIMAGE_H

#include <cublas_v2.h>
#include <string>
#include <gdal/gdal_priv.h>
#include <gdal/cpl_conv.h>

class GDALImage{

public:
    using size_t = std::size_t;

private:
    size_t _fileSize;
    int _height;
    int _width;
    void * _memPtr = NULL;

    int _pixelSize; //in bytes

    CPLVirtualMem * _poBandVirtualMem = NULL;
    GDALDataset * _poDataset = NULL;

public:
    GDALImage() = delete;
    GDALImage(std::string fn, int band=1, int cacheSizeInGB=0);

    void * getmemPtr()
    {
        return(_memPtr);
    }

    size_t getFileSize()
    {
        return (_fileSize);
    }

    size_t getHeight() {
        return (_height);
    }

    size_t getWidth()
    {
        return (_width);
    }

    int getPixelSize()
    {
        return _pixelSize;
    }

    void loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, cudaStream_t stream);
    ~GDALImage();

};

#endif //__GDALIMAGE_H
