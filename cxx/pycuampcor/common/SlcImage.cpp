#include "SlcImage.h"
#ifdef PYCUAMPCOR_WITH_HDF5
#include "Hdf5SlcImage.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
#include <iostream>
#include <stdexcept>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

bool SlcImage::parseHdf5Name(const std::string& name, std::string& file, std::string& dataset)
{
    const std::string prefix = "HDF5:";
    if (name.compare(0, prefix.size(), prefix) != 0)
        return false;
    const std::string rest = name.substr(prefix.size());
    size_t sep;
    if (!rest.empty() && rest[0] == '"') {
        // quoted file name
        const size_t quote = rest.find('"', 1);
        if (quote == std::string::npos || quote + 1 >= rest.size() || rest[quote + 1] != ':')
            throw std::invalid_argument("Invalid HDF5 image name " + name + "; expect HDF5:\"<file>\":<dataset>");
        file = rest.substr(1, quote - 1);
        sep = quote + 1;
    }
    else {
        // the dataset is an absolute path (from the last ":/"), or else after the last ':'
        sep = rest.rfind(":/");
        if (sep == std::string::npos)
            sep = rest.rfind(':');
        if (sep == std::string::npos || sep == 0)
            throw std::invalid_argument("Invalid HDF5 image name " + name + "; expect HDF5:<file>:<dataset>");
        file = rest.substr(0, sep);
    }
    dataset = rest.substr(sep + 1);
    // as an absolute path with a single leading '/'
    const size_t first = dataset.find_first_not_of('/');
    if (first == std::string::npos)
        throw std::invalid_argument("Invalid HDF5 image name " + name + "; no dataset is given");
    dataset = "/" + dataset.substr(first);
    return true;
}

bool SlcImage::hasHdf5()
{
#ifdef PYCUAMPCOR_WITH_HDF5
    return true;
#else
    return false;
#endif
}

std::unique_ptr<SlcImage> SlcImage::open(const std::string& filepath, size_t img_height, size_t img_width,
                                         size_t pixel_size, size_t buffer_size)
{
    std::string file, dataset;
    if (parseHdf5Name(filepath, file, dataset)) {
#ifdef PYCUAMPCOR_WITH_HDF5
        return std::make_unique<Hdf5SlcImage>(file, dataset, img_height, img_width, pixel_size, buffer_size);
#else
        throw std::runtime_error("Cannot read " + filepath + ": pycuampcor is built without HDF5 support");
#endif
    }
    return std::make_unique<MmapSlcImage>(filepath, img_height, img_width, pixel_size, buffer_size);
}

MmapSlcImage::MmapSlcImage(const std::string& filepath, size_t img_height, size_t img_width, size_t pixel_size, size_t buffer_size)
    : width(img_width), height(img_height), pixel_size(pixel_size), fd(-1), mapped_data(nullptr),
      mapped_offset(0), mapped_size(0)
{
    file_size = width * height * pixel_size;
    max_map_size = buffer_size*1024*1024*1024;
    page_size = sysconf(_SC_PAGE_SIZE);  // Get system page size

    // Open the file
    fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    // check the file is large enough for the given image size
    struct stat st;
    if (fstat(fd, &st) == 0 && (size_t)st.st_size < file_size) {
        close(fd);
        fd = -1;
        throw std::runtime_error("The file " + filepath + " (" + std::to_string(st.st_size)
            + " bytes) is smaller than the image size " + std::to_string(height) + " x "
            + std::to_string(width) + " x " + std::to_string(pixel_size) + " bytes");
    }
}

void MmapSlcImage::remapIfNeeded(size_t required_start, size_t required_end)
{

    if(required_start < mapped_offset || required_end > mapped_offset + mapped_size)
    {
        // out of range, remap
        // unmap first, if necessary
        if(mapped_data!=nullptr)
            munmap(mapped_data, mapped_size);
        // align new mapping offset
        // round to the page size
        mapped_offset = (required_start/page_size)*page_size;
        // compute the mapped size
        mapped_size = file_size - mapped_offset;
        // not to exceed the buffer size
        if (mapped_size > max_map_size) {
            mapped_size = max_map_size;
        }
        // the mapped region must cover the requested range
        if (required_end > mapped_offset + mapped_size) {
            mapped_data = nullptr;
            mapped_size = 0;
            throw std::runtime_error("The requested image tile exceeds the file size or the mmap buffer size;"
                " check the image size or increase mmapSize (in GB)");
        }
        // remap
        mapped_data = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, mapped_offset);
        if (mapped_data == MAP_FAILED) {
            mapped_data = nullptr;
            mapped_size = 0;
            throw std::runtime_error("Failed to mmap file at offset " + std::to_string(mapped_offset));
        }
    }
    // else - in range, do nothing
}


/// load a tile of data h_tile x w_tile from CPU (mmap) to the device (GPU) or work (CPU) memory
/// @param dArray pointer for array in device memory
/// @param h_offset Down/Height offset
/// @param w_offset Across/Width offset
/// @param h_tile Down/Height tile size
/// @param w_tile Across/Width tile size
/// @param stream CUDA stream for copying (not used for CPU)
void MmapSlcImage::loadToDevice(void *dArray, size_t h_offset, size_t w_offset, size_t h_tile, size_t w_tile, stream_t stream)
{
    size_t tileStartAddress = (h_offset*width + w_offset)*pixel_size;
    size_t tileLastAddress = ((h_offset+h_tile-1)*width + w_offset + w_tile)*pixel_size;

    // remapping changes the shared mapped region; serialize among workers
    std::lock_guard<std::mutex> lock(mutex);

    remapIfNeeded(tileStartAddress, tileLastAddress);

    char *startPtr = (char *)mapped_data ;
    startPtr += tileStartAddress - mapped_offset;

    // @note
    // We assume down/across directions as rows/cols. Therefore, SLC mmap and device array both use row major.
    backendCopyFromHost2D(dArray, w_tile*pixel_size, startPtr, width*pixel_size,
        w_tile*pixel_size, h_tile, stream);
}

MmapSlcImage::~MmapSlcImage()
{
    if (mapped_data!=nullptr) {
        munmap(mapped_data, mapped_size);
    }
    if (fd != -1) {
        close(fd);
    }
}
  	  
// end of file


} // namespace