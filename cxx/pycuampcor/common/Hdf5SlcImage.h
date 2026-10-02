// -*- c++ -*-
// file Hdf5SlcImage.h
// an image source for a 2D dataset in an HDF5 file

#ifndef __HDF5SLCIMAGE_H
#define __HDF5SLCIMAGE_H

#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <hdf5.h>

#include "SlcImage.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/// a 2D dataset (float32, or complex64 as a compound of two float32) in an HDF5 file
///
/// Chunked datasets without filters, or with deflate (gzip) preceded optionally by shuffle
/// (as in the NISAR products), are read directly: the chunks are located once when opening,
/// read from the file and decoded in parallel without calling the HDF5 library, and kept in
/// a cache of up to buffer_size GB shared by all workers. Each worker decodes the chunks missing
/// for its tile with up to max_threads threads (set by PYCUAMPCOR_HDF5_THREADS, default 8).
/// Other datasets (contiguous, other filters, a user block, a non-zero fill value) are read
/// with the HDF5 library (H5Dread).
class Hdf5SlcImage : public SlcImage {
public:
    Hdf5SlcImage(const std::string& file, const std::string& dataset,
                 size_t image_height, size_t image_width, size_t pixel_size, size_t buffer_size,
                 size_t max_threads = defaultMaxThreads());

    /// the default number of threads to decode chunks: PYCUAMPCOR_HDF5_THREADS if set, or 8
    static size_t defaultMaxThreads();
    ~Hdf5SlcImage() override;

    void loadToHost(void* host, size_t h_offset, size_t w_offset,
                    size_t h_tile, size_t w_tile) override;
    size_t pixelSize() const override { return pixel_size; }

    /// whether the chunks are read and decoded directly (true) or by the HDF5 library (false)
    bool directRead() const { return direct; }

private:
    using Chunk = std::shared_ptr<const std::vector<char>>;

    /// location and filters of a stored chunk
    struct ChunkInfo {
        haddr_t addr = HADDR_UNDEF;  ///< file offset; HADDR_UNDEF if the chunk is not allocated
        hsize_t size = 0;            ///< stored (compressed) size in bytes
        unsigned filterMask = 0;     ///< bit i set if filter i was skipped for this chunk
    };

    // image
    std::string filename;
    size_t height, width, pixel_size;

    // hdf5 handles
    hid_t fileId = H5I_INVALID_HID;
    hid_t datasetId = H5I_INVALID_HID;
    hid_t memType = H5I_INVALID_HID;

    // direct reading of chunks
    bool direct = false;
    int fd = -1;
    size_t chunkHeight = 0, chunkWidth = 0;   ///< chunk dimensions
    size_t nChunksDown = 0, nChunksAcross = 0; ///< number of chunks
    bool shuffle = false, deflate = false;     ///< the filter pipeline
    int shuffleIndex = -1, deflateIndex = -1;  ///< filter positions in the pipeline
    std::vector<ChunkInfo> chunkInfo;          ///< row-major nChunksDown x nChunksAcross

    // cache of decoded chunks, least recently used first out
    struct CacheEntry {
        std::shared_future<Chunk> data;
        std::list<size_t>::iterator lruPosition;
    };
    std::mutex cacheMutex;
    std::unordered_map<size_t, CacheEntry> cache;
    std::list<size_t> lru;   ///< most recently used at the front
    size_t cacheCapacity = 1; ///< in chunks
    size_t maxThreads = 1;    ///< threads to decode the chunks missing for a tile

    void inspectLayout(hid_t dcpl, size_t buffer_size);
    Chunk decodeChunk(size_t index) const;
    std::vector<Chunk> getChunks(const std::vector<size_t>& indices);
    void loadDirect(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile);
    void loadLibrary(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile);
};

} // namespace

#endif //__HDF5SLCIMAGE_H
