/**
 * @file  Arrays.cpp
 * @brief A batch of 2D arrays in the work memory (CPU): allocation, copies, output
 */

// dependencies
#include "Arrays.h"
#include "ArraysFile.h"
#include "float2.h"

#include <cstring>
#include <new>
#include <utility>

namespace pycuampcor::cpu {

/// the alignment (in bytes) of the work arrays, suitable for the SIMD instructions used by fftw
static constexpr size_t alignment = 64;

/**
 * Allocate aligned memory (posix_memalign is available on all POSIX platforms,
 * while std::aligned_alloc requires macOS >= 10.15)
 * @param bytes the size (at least alignment bytes are allocated); the memory is freed with std::free
 */
static void * alignedAlloc(size_t bytes)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, std::max(bytes, alignment)) != 0)
        throw std::bad_alloc();
    return ptr;
}

/**
 * Count arrays of height x width, allocated in the work memory
 */
template <typename T>
Arrays<T>::Arrays(int height, int width, int count)
{
    T *ptr = static_cast<T *>(alignedAlloc(
        std::size_t(count)*std::size_t(height)*std::size_t(width)*sizeof(T)));
    data = view_type(ptr, height, width, count);
}

/**
 * Free the memory
 */
template <typename T>
Arrays<T>::~Arrays()
{
    std::free(data.data_handle());
}

/**
 * The arrays of other, which is left empty
 */
template <typename T>
Arrays<T>::Arrays(Arrays &&other) noexcept
    : data(std::exchange(other.data, view_type()))
{
}

/**
 * Free the memory, and take the arrays of other, which is left empty
 */
template <typename T>
Arrays<T> &Arrays<T>::operator=(Arrays &&other) noexcept
{
    if (this != &other) {
        std::free(data.data_handle());
        data = std::exchange(other.data, view_type());
    }
    return *this;
}

/**
 * Set the elements to zero
 */
template <typename T>
void Arrays<T>::setZero(stream_t)
{
    std::memset(static_cast<void *>(data.data_handle()), 0, size()*sizeof(T));
}

/**
 * Copy size() elements from host
 */
template <typename T>
void Arrays<T>::copyFromHost(const T *host, stream_t)
{
    std::memcpy(static_cast<void *>(data.data_handle()), host, size()*sizeof(T));
}

/**
 * A copy of the elements
 */
template <typename T>
std::vector<T> Arrays<T>::toHost(stream_t) const
{
    return std::vector<T>(data.data_handle(), data.data_handle() + size());
}

/**
 * Write the arrays to a binary file
 */
template <typename T>
void Arrays<T>::outputToFile(const std::string &fn, stream_t) const
{
    writeArraysToFile(fn, view());
}

// instantiations
template class Arrays<float>;
template class Arrays<float2>;
template class Arrays<float3>;
template class Arrays<double>;
template class Arrays<double2>;
template class Arrays<double3>;
template class Arrays<int2>;
template class Arrays<int>;

} // namespace
// end of file
