/**
 * @file  cuArrays.cpp
 * @brief A batch of 2D arrays in the device memory (CUDA): allocation, transfers, output
 */

// dependencies
#include "cuArrays.h"
#include "ArraysFile.h"
#include "cudaError.h"
#include "float2.h"

#include <utility>

namespace pycuampcor::cuda {

/**
 * Count arrays of height x width, allocated in the device memory
 */
template <typename T>
cuArrays<T>::cuArrays(int height, int width, int count)
{
    T *ptr = nullptr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&ptr),
        std::size_t(count)*std::size_t(height)*std::size_t(width)*sizeof(T)));
    data = view_type(ptr, height, width, count);
}

/**
 * Free the device memory
 */
template <typename T>
cuArrays<T>::~cuArrays()
{
    if (data.data_handle())
        checkCudaErrors(cudaFree(data.data_handle()));
}

/**
 * The arrays of other, which is left empty
 */
template <typename T>
cuArrays<T>::cuArrays(cuArrays &&other) noexcept
    : data(std::exchange(other.data, view_type()))
{
}

/**
 * Free the device memory, and take the arrays of other, which is left empty
 */
template <typename T>
cuArrays<T> &cuArrays<T>::operator=(cuArrays &&other) noexcept
{
    if (this != &other) {
        if (data.data_handle())
            cudaFree(data.data_handle());
        data = std::exchange(other.data, view_type());
    }
    return *this;
}

/**
 * Set the elements to zero (queued on the stream)
 */
template <typename T>
void cuArrays<T>::setZero(cudaStream_t stream)
{
    checkCudaErrors(cudaMemsetAsync(data.data_handle(), 0, size()*sizeof(T), stream));
}

/**
 * Copy size() elements from the host memory (queued on the stream)
 * @note from pageable memory, the copy returns once host can be reused
 */
template <typename T>
void cuArrays<T>::copyFromHost(const T *host, cudaStream_t stream)
{
    checkCudaErrors(cudaMemcpyAsync(data.data_handle(), host, size()*sizeof(T),
        cudaMemcpyHostToDevice, stream));
}

/**
 * The elements, copied to the host memory once the work queued on the stream is done
 */
template <typename T>
std::vector<T> cuArrays<T>::toHost(cudaStream_t stream) const
{
    std::vector<T> host(size());
    checkCudaErrors(cudaMemcpyAsync(host.data(), data.data_handle(), size()*sizeof(T),
        cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    return host;
}

/**
 * Write the arrays to a binary file (copied to the host memory first)
 */
template <typename T>
void cuArrays<T>::outputToFile(const std::string &fn, cudaStream_t stream) const
{
    const std::vector<T> host = toHost(stream);
    writeArraysToFile(fn, ArraysView<const T>(host.data(), height(), width(), count()));
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
