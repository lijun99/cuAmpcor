/**
 * @file  SlcImage.h
 * @brief Image sources (tile loaders) for SLC images: a raw binary file (mmap) or an HDF5 dataset
 */

#pragma once

#include "forward.h"

namespace pycuampcor {

/// a source of image tiles, read to host memory (the chunk loaders of the cpu and cuda
/// backends then use them in place or copy them to the device)
class SlcImage {
public:
    virtual ~SlcImage() = default;

    /// the size of a pixel (as loaded) in bytes
    virtual size_t pixelSize() const = 0;

    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset) to host memory,
    /// row major with a pitch of w_tile pixels
    virtual void loadToHost(void* host, size_t h_offset, size_t w_offset,
                            size_t h_tile, size_t w_tile) = 0;

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
    /// open a raw binary file of image_height x image_width pixels of pixel_size bytes, row major
    /// @param buffersize the size of the mmap window, in GB
    MmapSlcImage(const std::string& fn, size_t image_height, size_t image_width, size_t pixel_size, size_t buffersize);
    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset) to host memory,
    /// row major with a pitch of w_tile pixels
    void loadToHost(void* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile) override;
    /// the size of a pixel in bytes
    size_t pixelSize() const override { return pixel_size; }
    /// unmap the window and close the file
    ~MmapSlcImage() override;

private:
    int fd;               ///< the file descriptor
    size_t file_size;     ///< the size of the image in bytes
    size_t pixel_size;    ///< the size of a pixel in bytes
    size_t height;        ///< the number of rows of the image (down, azimuth)
    size_t width;         ///< the number of columns of the image (across, range)

    void* mapped_data;    ///< the mapped window, nullptr if none
    size_t page_size;     ///< the system page size (the alignment of the window offset)
    size_t mapped_offset; ///< the offset of the window in the file, in bytes
    size_t mapped_size;   ///< the size of the window, in bytes
    size_t max_map_size;  ///< the maximum size of the window (the buffer size), in bytes

    std::mutex mutex;  ///< serialize remapping/loading among workers

    /// remap the window if the byte range [required_start, required_end) of the file is not in it
    void remapIfNeeded(size_t required_start, size_t required_end);
};

} // namespace
