#include "Hdf5SlcImage.h"

#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

namespace {

// the HDF5 library may be built without thread safety; serialize all calls into it
std::mutex &hdf5Mutex()
{
    static std::mutex mutex;
    return mutex;
}

// whether the type is a little-endian 32-bit float
bool isFloat32(hid_t type)
{
    return H5Tget_class(type) == H5T_FLOAT && H5Tget_size(type) == 4
        && H5Tget_order(type) == H5T_ORDER_LE && H5Tequal(type, H5T_IEEE_F32LE) > 0;
}

// whether the type is a compound of two float32 at offsets 0 and 4 (complex64)
bool isComplex64(hid_t type)
{
    if (H5Tget_class(type) != H5T_COMPOUND || H5Tget_size(type) != 8 || H5Tget_nmembers(type) != 2)
        return false;
    for (unsigned i = 0; i < 2; i++) {
        hid_t member = H5Tget_member_type(type, i);
        const bool ok = isFloat32(member) && H5Tget_member_offset(type, i) == 4 * i;
        H5Tclose(member);
        if (!ok) return false;
    }
    return true;
}

// read exactly n bytes at offset from a file descriptor
void preadAll(int fd, void *buffer, size_t n, off_t offset, const std::string &filename)
{
    char *p = static_cast<char *>(buffer);
    while (n > 0) {
        const ssize_t got = ::pread(fd, p, n, offset);
        if (got <= 0)
            throw std::runtime_error("Failed to read " + std::to_string(n) + " bytes at offset "
                + std::to_string(offset) + " from " + filename);
        p += got;
        n -= got;
        offset += got;
    }
}

// collect the stored chunks from H5Dchunk_iter
struct ChunkIterData {
    size_t chunkHeight, chunkWidth, nChunksAcross;
    void *infos; // std::vector<ChunkInfo> of the image, opaque here
};

} // namespace

Hdf5SlcImage::Hdf5SlcImage(const std::string& file, const std::string& dataset,
                           size_t image_height, size_t image_width, size_t pixel_size_, size_t buffer_size)
    : filename(file), height(image_height), width(image_width), pixel_size(pixel_size_)
{
    std::lock_guard<std::mutex> lock(hdf5Mutex());
    const std::string name = "HDF5 dataset " + dataset + " in " + file;
    hid_t dapl = H5I_INVALID_HID, space = H5I_INVALID_HID, type = H5I_INVALID_HID, dcpl = H5I_INVALID_HID;
    try {
        H5E_BEGIN_TRY {
            fileId = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        } H5E_END_TRY
        if (fileId < 0)
            throw std::runtime_error("Failed to open the HDF5 file " + file);
        // the library chunk cache (for reading with H5Dread)
        dapl = H5Pcreate(H5P_DATASET_ACCESS);
        H5Pset_chunk_cache(dapl, 12421, std::max<size_t>(buffer_size, 1) << 30, 1.0);
        H5E_BEGIN_TRY {
            datasetId = H5Dopen2(fileId, dataset.c_str(), dapl);
        } H5E_END_TRY
        if (datasetId < 0)
            throw std::runtime_error("Failed to open the " + name);

        // check the shape
        space = H5Dget_space(datasetId);
        hsize_t dims[2] = {0, 0};
        if (H5Sget_simple_extent_ndims(space) != 2)
            throw std::runtime_error("The " + name + " is not two-dimensional");
        H5Sget_simple_extent_dims(space, dims, nullptr);
        if (dims[0] != height || dims[1] != width)
            throw std::runtime_error("The " + name + " has the size " + std::to_string(dims[0]) + " x "
                + std::to_string(dims[1]) + ", not the given image size " + std::to_string(height)
                + " x " + std::to_string(width));

        // check the data type
        type = H5Dget_type(datasetId);
        const bool typeOk = (pixel_size == 4 && isFloat32(type)) || (pixel_size == 8 && isComplex64(type));
        if (!typeOk)
            throw std::runtime_error("The " + name + " is not of the expected type ("
                + (pixel_size == 8 ? std::string("complex64, a compound of two float32")
                                   : std::string("float32")) + ")");
        memType = H5Tget_native_type(type, H5T_DIR_ASCEND);

        // decide whether to read the chunks directly
        dcpl = H5Dget_create_plist(datasetId);
        inspectLayout(dcpl, buffer_size);
    }
    catch (...) {
        for (hid_t id : {dcpl, dapl}) if (id >= 0) H5Pclose(id);
        if (type >= 0) H5Tclose(type);
        if (space >= 0) H5Sclose(space);
        if (memType >= 0) H5Tclose(memType);
        if (datasetId >= 0) H5Dclose(datasetId);
        if (fileId >= 0) H5Fclose(fileId);
        if (fd >= 0) ::close(fd);
        throw;
    }
    H5Pclose(dcpl);
    H5Pclose(dapl);
    H5Tclose(type);
    H5Sclose(space);

    if (direct) {
        std::string filters = shuffle ? (deflate ? "shuffle+deflate" : "shuffle") : (deflate ? "deflate" : "none");
        std::cout << "  HDF5 " << dataset << ": chunks " << chunkHeight << " x " << chunkWidth
                  << ", filters " << filters << ", decoded directly (cache of " << cacheCapacity << " chunks)" << std::endl;
    }
    else {
        std::cout << "  HDF5 " << dataset << ": read with the HDF5 library" << std::endl;
    }
}

/// check whether the dataset can be read directly, and locate its chunks if so
void Hdf5SlcImage::inspectLayout(hid_t dcpl, size_t buffer_size)
{
    direct = false;
#if H5_VERSION_GE(1, 14, 0)
    if (H5Pget_layout(dcpl) != H5D_CHUNKED)
        return;
    // a user block shifts the file offsets of the chunks; leave it to the library
    hid_t fcpl = H5Fget_create_plist(fileId);
    hsize_t userblock = 0;
    H5Pget_userblock(fcpl, &userblock);
    H5Pclose(fcpl);
    if (userblock != 0)
        return;
    // unallocated chunks are read as zeros; leave a non-zero fill value to the library
    H5D_fill_value_t fillStatus;
    if (H5Pfill_value_defined(dcpl, &fillStatus) < 0)
        return;
    if (fillStatus == H5D_FILL_VALUE_USER_DEFINED) {
        std::vector<char> fill(pixel_size);
        if (H5Pget_fill_value(dcpl, memType, fill.data()) < 0
            || std::any_of(fill.begin(), fill.end(), [](char c) { return c != 0; }))
            return;
    }
    // the filters: none, deflate, or shuffle then deflate
    const int nFilters = H5Pget_nfilters(dcpl);
    for (int i = 0; i < nFilters; i++) {
        unsigned flags;
        size_t nValues = 0;
        const H5Z_filter_t filter = H5Pget_filter2(dcpl, i, &flags, &nValues, nullptr, 0, nullptr, nullptr);
        if (filter == H5Z_FILTER_SHUFFLE && !shuffle && !deflate) {
            shuffle = true;
            shuffleIndex = i;
        }
        else if (filter == H5Z_FILTER_DEFLATE && !deflate) {
            deflate = true;
            deflateIndex = i;
        }
        else
            return;
    }
    // the chunk grid
    hsize_t chunkDims[2];
    if (H5Pget_chunk(dcpl, 2, chunkDims) != 2)
        return;
    chunkHeight = chunkDims[0];
    chunkWidth = chunkDims[1];
    nChunksDown = (height + chunkHeight - 1) / chunkHeight;
    nChunksAcross = (width + chunkWidth - 1) / chunkWidth;
    // locate the stored chunks
    chunkInfo.assign(nChunksDown * nChunksAcross, ChunkInfo{});
    ChunkIterData data{chunkHeight, chunkWidth, nChunksAcross, &chunkInfo};
    auto collect = [](const hsize_t *offset, unsigned filterMask, haddr_t addr, hsize_t size, void *op) -> int {
        auto *d = static_cast<ChunkIterData *>(op);
        auto &infos = *static_cast<std::vector<ChunkInfo> *>(d->infos);
        const size_t index = (offset[0] / d->chunkHeight) * d->nChunksAcross + offset[1] / d->chunkWidth;
        if (index >= infos.size())
            return H5_ITER_ERROR;
        infos[index] = ChunkInfo{size > 0 ? addr : HADDR_UNDEF, size, filterMask};
        return H5_ITER_CONT;
    };
    herr_t status;
    H5E_BEGIN_TRY {
        status = H5Dchunk_iter(datasetId, H5P_DEFAULT, collect, &data);
    } H5E_END_TRY
    if (status < 0) {
        chunkInfo.clear();
        return;
    }
    fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) {
        chunkInfo.clear();
        return;
    }
    // cache: the buffer size in GB, at least one row of chunks
    const size_t chunkBytes = chunkHeight * chunkWidth * pixel_size;
    cacheCapacity = std::max((buffer_size << 30) / chunkBytes, nChunksAcross);
    direct = true;
#else
    (void)dcpl;
    (void)buffer_size;
#endif
}

Hdf5SlcImage::~Hdf5SlcImage()
{
    if (fd >= 0) ::close(fd);
    std::lock_guard<std::mutex> lock(hdf5Mutex());
    if (memType >= 0) H5Tclose(memType);
    if (datasetId >= 0) H5Dclose(datasetId);
    if (fileId >= 0) H5Fclose(fileId);
}

/// read a chunk from the file and undo its filters (in the reverse order of the pipeline)
Hdf5SlcImage::Chunk Hdf5SlcImage::decodeChunk(size_t index) const
{
    const size_t chunkBytes = chunkHeight * chunkWidth * pixel_size;
    auto out = std::make_shared<std::vector<char>>(chunkBytes);
    const ChunkInfo &info = chunkInfo[index];
    if (info.addr == HADDR_UNDEF)
        return out; // not allocated: zeros

    std::vector<char> stored(info.size);
    preadAll(fd, stored.data(), info.size, static_cast<off_t>(info.addr), filename);

    const bool inflate = deflate && !(info.filterMask & (1u << deflateIndex));
    const bool unshuffle = shuffle && !(info.filterMask & (1u << shuffleIndex));
    std::vector<char> inflated;
    const std::vector<char> *data = &stored;
    if (inflate) {
        inflated.resize(chunkBytes);
        uLongf length = chunkBytes;
        const int status = ::uncompress(reinterpret_cast<Bytef *>(inflated.data()), &length,
            reinterpret_cast<const Bytef *>(stored.data()), stored.size());
        if (status != Z_OK || length != chunkBytes)
            throw std::runtime_error("Failed to decompress a chunk of " + filename);
        data = &inflated;
    }
    if (data->size() != chunkBytes)
        throw std::runtime_error("Unexpected chunk size in " + filename);
    if (unshuffle) {
        // the shuffle filter stores byte j of all elements together: shuffled[j*n + i] = element i byte j
        const size_t n = chunkBytes / pixel_size;
        const char *src = data->data();
        char *dst = out->data();
        for (size_t i = 0; i < n; i++)
            for (size_t j = 0; j < pixel_size; j++)
                dst[i * pixel_size + j] = src[j * n + i];
    }
    else {
        std::memcpy(out->data(), data->data(), chunkBytes);
    }
    return out;
}

/// get the decoded chunks from the cache, decoding the missing ones in parallel
std::vector<Hdf5SlcImage::Chunk> Hdf5SlcImage::getChunks(const std::vector<size_t>& indices)
{
    std::vector<std::shared_future<Chunk>> futures;
    std::vector<std::pair<size_t, std::promise<Chunk>>> missing;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        for (size_t index : indices) {
            auto found = cache.find(index);
            if (found != cache.end()) {
                // mark as the most recently used
                lru.splice(lru.begin(), lru, found->second.lruPosition);
                futures.push_back(found->second.data);
                continue;
            }
            // decoded by this call; others needing it wait for the future
            std::promise<Chunk> promise;
            std::shared_future<Chunk> future = promise.get_future().share();
            lru.push_front(index);
            cache.emplace(index, CacheEntry{future, lru.begin()});
            futures.push_back(future);
            missing.emplace_back(index, std::move(promise));
        }
        // evict the least recently used (the chunks needed now are kept alive by their futures)
        while (cache.size() > cacheCapacity) {
            cache.erase(lru.back());
            lru.pop_back();
        }
    }

    // decode the missing chunks, in parallel
    auto decode = [this](std::pair<size_t, std::promise<Chunk>> &task) {
        try {
            task.second.set_value(decodeChunk(task.first));
        }
        catch (...) {
            task.second.set_exception(std::current_exception());
        }
    };
    const size_t nThreads = std::min<size_t>(missing.size(), std::max(1u, std::thread::hardware_concurrency()));
    if (nThreads <= 1) {
        for (auto &task : missing) decode(task);
    }
    else {
        std::vector<std::thread> threads;
        for (size_t t = 0; t < nThreads; t++)
            threads.emplace_back([&, t]() {
                for (size_t k = t; k < missing.size(); k += nThreads) decode(missing[k]);
            });
        for (auto &thread : threads) thread.join();
    }

    std::vector<Chunk> chunks;
    for (auto &future : futures) chunks.push_back(future.get());
    return chunks;
}

/// assemble a tile from the decoded chunks
void Hdf5SlcImage::loadDirect(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile)
{
    const size_t r0 = h_offset / chunkHeight, r1 = (h_offset + h_tile - 1) / chunkHeight;
    const size_t c0 = w_offset / chunkWidth, c1 = (w_offset + w_tile - 1) / chunkWidth;
    std::vector<size_t> indices;
    for (size_t r = r0; r <= r1; r++)
        for (size_t c = c0; c <= c1; c++)
            indices.push_back(r * nChunksAcross + c);
    const std::vector<Chunk> chunks = getChunks(indices);

    size_t k = 0;
    for (size_t r = r0; r <= r1; r++) {
        for (size_t c = c0; c <= c1; c++, k++) {
            const char *chunk = chunks[k]->data();
            // the part of the tile in this chunk, in image pixels
            const size_t rowStart = std::max(h_offset, r * chunkHeight);
            const size_t rowEnd = std::min(h_offset + h_tile, (r + 1) * chunkHeight);
            const size_t colStart = std::max(w_offset, c * chunkWidth);
            const size_t colEnd = std::min(w_offset + w_tile, (c + 1) * chunkWidth);
            const size_t bytes = (colEnd - colStart) * pixel_size;
            for (size_t row = rowStart; row < rowEnd; row++)
                std::memcpy(host + ((row - h_offset) * w_tile + (colStart - w_offset)) * pixel_size,
                            chunk + ((row - r * chunkHeight) * chunkWidth + (colStart - c * chunkWidth)) * pixel_size,
                            bytes);
        }
    }
}

/// read a tile with the HDF5 library
void Hdf5SlcImage::loadLibrary(char* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile)
{
    std::lock_guard<std::mutex> lock(hdf5Mutex());
    const hsize_t start[2] = {h_offset, w_offset};
    const hsize_t count[2] = {h_tile, w_tile};
    hid_t fileSpace = H5Dget_space(datasetId);
    hid_t memSpace = H5Screate_simple(2, count, nullptr);
    herr_t status = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    if (status >= 0)
        status = H5Dread(datasetId, memType, memSpace, fileSpace, H5P_DEFAULT, host);
    H5Sclose(memSpace);
    H5Sclose(fileSpace);
    if (status < 0)
        throw std::runtime_error("Failed to read a tile from " + filename);
}

/// load a tile of h_tile x w_tile pixels to host memory
void Hdf5SlcImage::loadToHost(void* host, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile)
{
    if (h_tile == 0 || w_tile == 0) return;
    if (h_offset + h_tile > height || w_offset + w_tile > width)
        throw std::runtime_error("The requested tile exceeds the image " + filename);
    if (direct)
        loadDirect(static_cast<char *>(host), h_offset, w_offset, h_tile, w_tile);
    else
        loadLibrary(static_cast<char *>(host), h_offset, w_offset, h_tile, w_tile);
}

} // namespace
// end of file
