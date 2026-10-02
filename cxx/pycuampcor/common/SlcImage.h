// -*- c++ -*-
// file slcimage.h
// image sources (tile loaders) for slc images: a raw binary file (mmap) or an HDF5 dataset

#ifndef __SLCIMAGE_H
#define __SLCIMAGE_H

#include <memory>
#include <string>
#include <mutex>
#include "backend.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/// a source of image tiles, loaded to the device (GPU) or work (CPU) memory
class SlcImage {
public:
    virtual ~SlcImage() = default;

    /// the size of a pixel (as loaded) in bytes
    virtual size_t pixelSize() const = 0;

    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset) to host memory,
    /// row major with a pitch of w_tile pixels
    virtual void loadToHost(void* host, size_t h_offset, size_t w_offset,
                            size_t h_tile, size_t w_tile) = 0;

    /// load a tile to the device (GPU) or work (CPU) memory, through a pageable host buffer
    void loadToDevice(void* dArray, size_t h_offset, size_t w_offset,
                      size_t h_tile, size_t w_tile, stream_t stream);

    /// open an image of image_height x image_width pixels of pixel_size bytes
    /// @param fn the image name: a raw binary file (row major), or a 2D dataset in an HDF5 file
    ///     as HDF5:<file>:<dataset> (as in GDAL; the file may be quoted, HDF5:"<file>":<dataset>)
    /// @param reader how to read the image: "raw" (a raw binary file), "hdf5" (an HDF5 dataset;
    ///     the HDF5: prefix of the name is optional), or "auto" (hdf5 if the name starts with HDF5:,
    ///     raw otherwise)
    /// @param buffer_size the host memory buffer for reading the image, in GB
    ///     (the mmap window for a raw file, the cache of decoded chunks for HDF5)
    static std::unique_ptr<SlcImage> open(const std::string& fn, const std::string& reader,
                                          size_t image_height, size_t image_width,
                                          size_t pixel_size, size_t buffer_size);

    /// split an image name HDF5:<file>:<dataset> into the file and the dataset
    /// @return false if the name is not of an HDF5 dataset
    static bool parseHdf5Name(const std::string& name, std::string& file, std::string& dataset);

    /// whether HDF5 datasets are supported (built with HDF5)
    static bool hasHdf5();
};

/// a raw binary image file, memory mapped in a window of up to buffer_size GB
class MmapSlcImage : public SlcImage {
public:
    // disable default constructor
    MmapSlcImage()=delete;
    // constructor
    MmapSlcImage(const std::string& fn, size_t image_height, size_t image_width, size_t pixel_size, size_t buffersize);
    // interface
    void loadToHost(void* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile) override;
    size_t pixelSize() const override { return pixel_size; }
    // destructor
    ~MmapSlcImage() override;

private:
    int fd;
    size_t file_size;
    size_t pixel_size;
    size_t height;
    size_t width;

    void* mapped_data;
    size_t page_size;
    size_t mapped_offset;
    size_t mapped_size;
    size_t max_map_size;

    std::mutex mutex;  ///< serialize remapping/loading among workers

    void remapIfNeeded(size_t required_start, size_t required_end);
};

} // namespace

#endif //__SLCIMAGE_H
