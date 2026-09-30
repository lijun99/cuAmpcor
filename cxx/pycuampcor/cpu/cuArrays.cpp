/**
 * \file  cuArrays.cpp
 * \brief  Implementations for cuArrays class (CPU backend)
 *
 */

// dependencies
#include "cuArrays.h"
#include "float2.h"
#include "data_types.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>

namespace pycuampcor::cpu {

// alignment for work arrays, suitable for SIMD instructions used by fftw
static constexpr size_t alignment = 64;

// allocate aligned memory (posix_memalign is available on all POSIX platforms,
// while std::aligned_alloc requires macOS >= 10.15)
static void * alignedAlloc(size_t bytes)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, std::max(bytes, alignment)) != 0)
        throw std::bad_alloc();
    return ptr;
}

// allocate arrays in (device) work memory
template <typename T>
void cuArrays<T>::allocate()
{
    devData = (T *)alignedAlloc(getByteSize());
    is_allocated = 1;
}

// allocate arrays in host memory
template <typename T>
void cuArrays<T>::allocateHost()
{
    hostData = (T *)alignedAlloc(getByteSize());
    is_allocatedHost = 1;
}

// deallocate arrays in (device) work memory
template <typename T>
void cuArrays<T>::deallocate()
{
    std::free(devData);
    devData = nullptr;
    is_allocated = 0;
}

// deallocate arrays in host memory
template <typename T>
void cuArrays<T>::deallocateHost()
{
    std::free(hostData);
    hostData = nullptr;
    is_allocatedHost = 0;
}

// copy arrays from (device) work memory to host
template <typename T>
void cuArrays<T>::copyToHost(stream_t)
{
    std::memcpy(hostData, devData, getByteSize());
}

// copy arrays from host to (device) work memory
template <typename T>
void cuArrays<T>::copyToDevice(stream_t)
{
    std::memcpy(devData, hostData, getByteSize());
}

// set to 0
template <typename T>
void cuArrays<T>::setZero(stream_t)
{
    std::memset((void *)devData, 0, getByteSize());
}

// Overloaded << operator for composite type
std::ostream& operator<<(std::ostream& os, const float2& p) {
        return os << "(" << p.x << ", " << p.y << ")";
}

std::ostream& operator<<(std::ostream& os, const float3& p) {
        return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator<<(std::ostream& os, const double2& p) {
        return os << "(" << p.x << ", " << p.y << ")";
}

std::ostream& operator<<(std::ostream& os, const double3& p) {
        return os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator<<(std::ostream& os, const int2& p) {
        return os << "(" << p.x << ", " << p.y << ")";
}

// output (partial) data when debugging
template <typename T>
void cuArrays<T>::debuginfo(stream_t stream) {
    // output size info
    std::cout << "Image height,width,count: " << height << "," << width << "," << count << std::endl;
    // check whether host data is allocated
    if( !is_allocatedHost)
        allocateHost();
    // copy to host
    copyToHost(stream);

    // set a max output range
    int range = std::min(10, size*count);
    // first 10 data
    for(int i=0; i<range; i++)
        std::cout << "(" <<hostData[i]  << ")" ;
    std::cout << std::endl;
    // last 10 data
    if(size*count>range) {
        for(int i=size*count-range; i<size*count; i++)
            std::cout << "(" <<hostData[i] << ")" ;
        std::cout << std::endl;
    }
}

// output to file by copying to host at first
template<typename T>
void cuArrays<T>::outputToFile(std::string fn, stream_t stream)
{
    if( !is_allocatedHost)
        allocateHost();
    copyToHost(stream);
    outputHostToFile(fn);
}

// save the host data to (binary) file
template <typename T>
void cuArrays<T>::outputHostToFile(std::string fn)
{
    std::ofstream file;
    file.open(fn.c_str(),  std::ios_base::binary);
    file.write((char *)hostData, getByteSize());
    file.close();
    std::cout << fn << " size: height " << height << " width "<< width << " count " << count << "\n";
}

// instantiations
template class cuArrays<float>;
template class cuArrays<float2>;
template class cuArrays<float3>;
template class cuArrays<double>;
template class cuArrays<double2>;
template class cuArrays<double3>;
template class cuArrays<int2>;
template class cuArrays<int>;

} // namespace
// end of file
