/**
 * @file  Hdf5SlcImage.h
 * @brief An image source for a 2D dataset in an HDF5 file
 */

#pragma once

#include "SlcImage.h"

namespace pycuampcor {

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
    /// open a 2D dataset of an HDF5 file as an image of image_height x image_width pixels
    /// @param file,dataset the HDF5 file and the path of the dataset in it
    /// @param pixel_size the size of a pixel in bytes: 4 (float32) or 8 (complex64)
    /// @param buffer_size the cache of decoded chunks (and the HDF5 library chunk cache), in GB
    /// @param max_threads the maximum number of threads to decode the chunks missing for a tile
    Hdf5SlcImage(const std::string& file, const std::string& dataset,
                 size_t image_height, size_t image_width, size_t pixel_size, size_t buffer_size,
                 size_t max_threads = defaultMaxThreads());

    /// the default number of threads to decode chunks: PYCUAMPCOR_HDF5_THREADS if set, or 8
    static size_t defaultMaxThreads();
    /// close the dataset and the file
    ~Hdf5SlcImage() override;

    /// load a tile of h_tile x w_tile pixels starting at (h_offset, w_offset) to host memory,
    /// row major with a pitch of w_tile pixels
    void loadToHost(void* host, size_t h_offset, size_t w_offset,
                    size_t h_tile, size_t w_tile) override;
    /// the size of a pixel in bytes: 4 (float32) or 8 (complex64)
    size_t pixelSize() const override { return pixel_size; }

    /// whether the chunks are read and decoded directly (true) or by the HDF5 library (false)
    bool directRead() const { return direct; }

private:
    /// a decoded chunk, chunkHeight x chunkWidth pixels row major, shared by the cache and the
    /// readers of the tiles
    using Chunk = std::shared_ptr<const std::vector<char>>;

    /// location and filters of a stored chunk
    struct ChunkInfo {
        haddr_t addr = HADDR_UNDEF;  ///< file offset; HADDR_UNDEF if the chunk is not allocated
        hsize_t size = 0;            ///< stored (compressed) size in bytes
        unsigned filterMask = 0;     ///< bit i set if filter i was skipped for this chunk
    };

    std::string filename;                      ///< the HDF5 file
    size_t height, width, pixel_size;          ///< the image size in pixels, the pixel size in bytes

    hid_t fileId = H5I_INVALID_HID;            ///< the HDF5 handle of the file
    hid_t datasetId = H5I_INVALID_HID;         ///< the HDF5 handle of the dataset
    hid_t memType = H5I_INVALID_HID;           ///< the native (memory) type of the pixels

    bool direct = false;                       ///< whether the chunks are read and decoded directly
    int fd = -1;                               ///< the file descriptor (direct reads)
    size_t chunkHeight = 0, chunkWidth = 0;   ///< chunk dimensions
    size_t nChunksDown = 0, nChunksAcross = 0; ///< number of chunks
    bool shuffle = false, deflate = false;     ///< the filter pipeline
    int shuffleIndex = -1, deflateIndex = -1;  ///< filter positions in the pipeline
    std::vector<ChunkInfo> chunkInfo;          ///< row-major nChunksDown x nChunksAcross

    /// an entry of the cache of decoded chunks (least recently used first out)
    struct CacheEntry {
        std::shared_future<Chunk> data;           ///< the chunk, ready once decoded
        std::list<size_t>::iterator lruPosition;  ///< the position of the chunk index in lru
    };
    std::mutex cacheMutex;                         ///< serialize the access to the cache among workers
    std::unordered_map<size_t, CacheEntry> cache;  ///< the decoded chunks by their index
    std::list<size_t> lru;   ///< most recently used at the front
    size_t cacheCapacity = 1; ///< in chunks
    size_t maxThreads = 1;    ///< threads to decode the chunks missing for a tile

    /// check whether the dataset can be read directly, and locate its chunks and size the cache
    /// if so
    /// @param dcpl the dataset creation property list (the layout and the filters)
    /// @param buffer_size the cache of decoded chunks, in GB
    void inspectLayout(hid_t dcpl, size_t buffer_size);
    /// read a chunk (row-major index in the chunk grid) from the file and undo its filters
    Chunk decodeChunk(size_t index) const;
    /// get the decoded chunks from the cache, decoding the missing ones in parallel
    std::vector<Chunk> getChunks(const std::vector<size_t>& indices);
    /// load a tile to host memory, assembled from the decoded chunks
    void loadDirect(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile);
    /// load a tile to host memory with the HDF5 library (H5Dread)
    void loadLibrary(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile);
};

} // namespace
