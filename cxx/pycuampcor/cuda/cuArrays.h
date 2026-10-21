/**
 * @file  cuArrays.h
 * @brief Header file for cuArrays class
 *
 * A class describes a batch of images (in 2d arrays).
 * Each image has size (height, width)
 * The number of images (countH, countW) or (1, count).
 **/

#pragma once

#include "forward.h"
#include "ArraysView.h"

// cuda dependencies

namespace pycuampcor::cuda {

template <typename T>
class cuArrays{

public:
    int height; ///< x, row, down, length, azimuth, along the track
    int width;  // y, col, across, range, along the sight
    int size;   // one image size, height*width
    int countH; // number of images along height direction
    int countW; // number of images along width direction
    int count;  // countW*countH, number of images

    bool is_allocated; // whether the data is allocated in device memory
    bool is_allocatedHost; // whether the data is allocated in host memory

    T* devData; // pointer to data in device (gpu) memory
    T* hostData; // pointer to data in host (cpu) memory

    // default constructor, empty
    cuArrays() : height(0), width(0), size(0), countH(0), countW(0), count(0),
        is_allocated(0), is_allocatedHost(0),
        devData(nullptr), hostData(nullptr) {}

    // constructor for single image
    cuArrays(size_t h, size_t w) : height(h), width(w), size(h*w), countH(1), countW(1), count(1),
        is_allocated(0), is_allocatedHost(0),
        devData(nullptr), hostData(nullptr) {}

    // constructor for multiple images with a total count
    cuArrays(size_t h, size_t w, size_t n) : height(h), width(w), size(h*w), countH(1), countW(n), count(n),
        is_allocated(0), is_allocatedHost(0),
        devData(nullptr), hostData(nullptr) {}

    // constructor for multiple images with (countH, countW)
    cuArrays(size_t h, size_t w, size_t ch, size_t cw) : height(h), width(w), size(h*w), countH(ch), countW(cw), count(ch*cw),
        is_allocated(0), is_allocatedHost(0),
        devData(nullptr), hostData(nullptr) {}

    // disable copying
    cuArrays(const cuArrays&) = delete;
    cuArrays& operator=(const cuArrays&) = delete;

    // memory allocation
    void allocate();
    void allocateHost();
    void deallocate();
    void deallocateHost();

    // copy data between device and host memories
    void copyToHost(cudaStream_t stream);
    void copyToDevice(cudaStream_t stream);

    // get the total size
    size_t getSize()
    {
        return size*count;
    }

    // get the total size in byte
    inline long getByteSize()
    {
        return width*height*count*sizeof(T);
    }

    // destructor
    ~cuArrays()
    {
        if(is_allocated)
            deallocate();
        if(is_allocatedHost)
            deallocateHost();
    }

    /// a view of the data in the device memory
    ArraysView<T> view() { return ArraysView<T>(devData, height, width, count); }
    /// a read-only view of the data in the device memory
    ArraysView<const T> view() const { return ArraysView<const T>(devData, height, width, count); }
    /// the arrays as a view, e.g., for the kernels
    operator ArraysView<T>() { return view(); }
    /// the arrays as a read-only view
    operator ArraysView<const T>() const { return view(); }

    // set zeroes
    void setZero(cudaStream_t stream);
    // output when debugging
    void debuginfo(cudaStream_t stream) ;
    void debuginfo(cudaStream_t stream, float factor);
    // write to files
    void outputToFile(std::string fn, cudaStream_t stream);
    void outputHostToFile(std::string fn);

};

std::ostream& operator<<(std::ostream& os, const float2& p);
std::ostream& operator<<(std::ostream& os, const float3& p);
std::ostream& operator<<(std::ostream& os, const double2& p);
std::ostream& operator<<(std::ostream& os, const float3& p);
std::ostream& operator<<(std::ostream& os, const int2& p);

} // namespace
//end of file
