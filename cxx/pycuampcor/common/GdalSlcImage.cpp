/**
 * @file  GdalSlcImage.cpp
 * @brief An image source for a raster read with GDAL
 */

#include "GdalSlcImage.h"

namespace pycuampcor {

namespace {

/// register the GDAL drivers, once
void registerGdal()
{
    static std::once_flag once;
    std::call_once(once, [] { GDALAllRegister(); });
}

/// keep the GDAL errors of this thread quiet (they are raised as exceptions instead) while in scope
struct QuietGdalErrors {
    QuietGdalErrors() { CPLPushErrorHandler(CPLQuietErrorHandler); CPLErrorReset(); }  ///< quiet
    ~QuietGdalErrors() { CPLPopErrorHandler(); }                                       ///< restore
    /// the message of the last error, or "" if none
    static std::string last()
    {
        const char* msg = CPLGetLastErrorMsg();
        return (msg && *msg) ? std::string(": ") + msg : std::string();
    }
};

} // namespace

/**
 * Open the first band of a raster, checking its size and its type (real or complex) against the
 * image
 */
GdalSlcImage::GdalSlcImage(const std::string& fn, size_t image_height, size_t image_width,
                           size_t pixel_size_)
    : filename(fn), height(image_height), width(image_width), pixel_size(pixel_size_)
{
    if (pixel_size != 4 && pixel_size != 8)
        throw std::invalid_argument("Unsupported pixel size " + std::to_string(pixel_size)
            + " for the GDAL raster " + filename + "; expect 4 (float32) or 8 (complex64)");
    registerGdal();
    QuietGdalErrors quiet;
    dataset = GDALOpenEx(filename.c_str(), GDAL_OF_RASTER | GDAL_OF_READONLY, nullptr, nullptr, nullptr);
    if (dataset == nullptr)
        throw std::runtime_error("Failed to open the raster " + filename + " with GDAL" + QuietGdalErrors::last());
    try {
        if (GDALGetRasterCount(dataset) < 1)
            throw std::runtime_error("The raster " + filename + " has no band");
        band = GDALGetRasterBand(dataset, 1);
        // check the size
        const size_t rows = GDALGetRasterYSize(dataset), cols = GDALGetRasterXSize(dataset);
        if (rows != height || cols != width)
            throw std::runtime_error("The raster " + filename + " has the size " + std::to_string(rows)
                + " x " + std::to_string(cols) + ", not the given image size " + std::to_string(height)
                + " x " + std::to_string(width));
        // check the type: complex for complex64 images, real for float32 images
        const GDALDataType type = GDALGetRasterDataType(band);
        const bool complex = GDALDataTypeIsComplex(type);
        if (complex != (pixel_size == 8))
            throw std::runtime_error("The raster " + filename + " is " + GDALGetDataTypeName(type)
                + ", " + (complex ? "complex" : "real") + ", but the image type is "
                + (pixel_size == 8 ? "complex64" : "float32"));
    }
    catch (...) {
        GDALClose(dataset);
        throw;
    }
}

/**
 * Close the raster
 */
GdalSlcImage::~GdalSlcImage()
{
    if (dataset != nullptr)
        GDALClose(dataset);
}

/**
 * Load a tile of h_tile x w_tile pixels to host memory, converted by GDAL to float32 or complex64
 * @param host the tile in host memory, row major with a pitch of w_tile pixels
 * @param h_offset,w_offset the first pixel of the tile (down, across)
 * @param h_tile,w_tile the size of the tile (down, across)
 */
void GdalSlcImage::loadToHost(void* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile)
{
    if (h_offset + h_tile > height || w_offset + w_tile > width)
        throw std::runtime_error("The requested image tile exceeds the raster " + filename
            + "; check the image size");
    const GDALDataType bufferType = (pixel_size == 8) ? GDT_CFloat32 : GDT_Float32;

    // a GDAL dataset is not thread-safe; serialize among workers
    std::lock_guard<std::mutex> lock(mutex);
    QuietGdalErrors quiet;
    const CPLErr err = GDALRasterIO(band, GF_Read,
        static_cast<int>(w_offset), static_cast<int>(h_offset), static_cast<int>(w_tile), static_cast<int>(h_tile),
        host, static_cast<int>(w_tile), static_cast<int>(h_tile), bufferType, 0, 0);
    if (err != CE_None)
        throw std::runtime_error("Failed to read the raster " + filename + " with GDAL" + QuietGdalErrors::last());
}

} // namespace
// end of file
