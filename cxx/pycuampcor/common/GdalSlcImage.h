/**
 * @file  GdalSlcImage.h
 * @brief An image source for a raster read with GDAL (GeoTIFF, VRT, ...)
 */

#pragma once

#include "SlcImage.h"

namespace pycuampcor {

/// the first band of a raster read with GDAL (GeoTIFF, VRT, and the other formats of GDAL)
///
/// The pixels are converted by GDAL to the type of the image: a real band of any type (e.g.,
/// Byte, UInt16, Int16, Float32, Float64) to float32, a complex band of any type (e.g., CInt16,
/// CFloat32) to complex64. The reads are serialized among the workers (a GDAL dataset is not
/// thread-safe); GDAL caches the blocks read (GDAL_CACHEMAX).
class GdalSlcImage : public SlcImage {
public:
    /// open the first band of a raster of image_height x image_width pixels
    /// @param fn the raster: a file, or a dataset name of GDAL
    /// @param pixel_size the size of a pixel as loaded: 4 (float32, from a real band) or 8
    ///     (complex64, from a complex band)
    GdalSlcImage(const std::string& fn, size_t image_height, size_t image_width, size_t pixel_size);
    /// close the raster
    ~GdalSlcImage() override;

    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset) to host memory,
    /// row major with a pitch of w_tile pixels
    void loadToHost(void* host, size_t h_offset, size_t w_offset,
                    size_t h_tile, size_t w_tile) override;
    /// the size of a pixel as loaded in bytes: 4 (float32) or 8 (complex64)
    size_t pixelSize() const override { return pixel_size; }

private:
    std::string filename;                ///< the raster
    size_t height, width, pixel_size;    ///< the image size in pixels, the pixel size in bytes
    GDALDatasetH dataset = nullptr;      ///< the GDAL handle of the raster
    GDALRasterBandH band = nullptr;      ///< the GDAL handle of the band read
    std::mutex mutex;                    ///< serialize the reads among workers
};

} // namespace
