// -*- c++ -*-
// file slcimage.h
// image sources (tile loaders) for slc images: a raw binary file (mmap)

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

    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset), row major
    virtual void loadToDevice(void* dArray, size_t h_offset, size_t w_offset,
                              size_t h_tile, size_t w_tile, stream_t stream) = 0;

    /// open an image file of image_height x image_width pixels of pixel_size bytes
    /// @param fn the file name, a raw binary file (row major)
    /// @param buffer_size the host memory buffer for reading the image, in GB
    static std::unique_ptr<SlcImage> open(const std::string& fn, size_t image_height, size_t image_width,
                                          size_t pixel_size, size_t buffer_size);
};

/// a raw binary image file, memory mapped in a window of up to buffer_size GB
class MmapSlcImage : public SlcImage {
public:
    // disable default constructor
    MmapSlcImage()=delete;
    // constructor
    MmapSlcImage(const std::string& fn, size_t image_height, size_t image_width, size_t pixel_size, size_t buffersize);
    // interface
    void loadToDevice(void* dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, stream_t stream) override;
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
