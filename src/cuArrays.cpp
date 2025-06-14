/**
 * \file  cuArrays.cu
 * \brief  Implementations for cuArrays class
 *
 */

// dependencies
#include "cuArrays.h"
#include "cudaError.h"
#include "float2.h"
#include "data_types.h"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

// allocate arrays in device memory
template <typename T>
void cuArrays<T>::allocate()
{
    checkCudaErrors(cudaMalloc((void **)&devData, getByteSize()));
    is_allocated = 1;
}

// allocate arrays in host memory
template <typename T>
void cuArrays<T>::allocateHost()
{
    hostData = (T *)malloc(getByteSize());
    is_allocatedHost = 1;
}

// deallocate arrays in device memory
template <typename T>
void cuArrays<T>::deallocate()
{
    checkCudaErrors(cudaFree(devData));
    is_allocated = 0;
}

// deallocate arrays in host memory
template <typename T>
void cuArrays<T>::deallocateHost()
{
    free(hostData);
    is_allocatedHost = 0;
}

// copy arrays from device to host
// use asynchronous for possible overlaps between data copying and kernel execution
template <typename T>
void cuArrays<T>::copyToHost(cudaStream_t stream)
{
    checkCudaErrors(cudaMemcpyAsync(hostData, devData, getByteSize(), cudaMemcpyDeviceToHost, stream));
}

// copy arrays from host to device
template <typename T>
void cuArrays<T>::copyToDevice(cudaStream_t stream)
{
    checkCudaErrors(cudaMemcpyAsync(devData, hostData, getByteSize(), cudaMemcpyHostToDevice, stream));
}

// set to 0
template <typename T>
void cuArrays<T>::setZero(cudaStream_t stream)
{
    checkCudaErrors(cudaMemsetAsync(devData, 0, getByteSize(), stream));
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
void cuArrays<T>::debuginfo(cudaStream_t stream) {
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
void cuArrays<T>::outputToFile(std::string fn, cudaStream_t stream)
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
    std::cout << fn << "size: height " << height << " width "<< width << " count " << count << "\n";
}

// instantiations, required by python extensions
template class cuArrays<float>;
template class cuArrays<float2>;
template class cuArrays<float3>;
template class cuArrays<double>;
template class cuArrays<double2>;
template class cuArrays<double3>;
template class cuArrays<int2>;
template class cuArrays<int>;

// end of file
