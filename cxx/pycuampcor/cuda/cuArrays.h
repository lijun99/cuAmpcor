/**
 * @file  cuArrays.h
 * @brief A batch of 2D arrays in the device memory (CUDA)
 *
 * cuArrays owns count arrays of height x width (row-major, one after the other) in the device
 * memory: allocated when constructed, freed when destroyed. It can be moved, not copied. The
 * kernels and workers take views of the arrays (ArraysView.h), to which the arrays convert.
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cuda {

/// count 2D arrays of height x width of T in the device memory
template <typename T>
class cuArrays {
public:
    using value_type = T;                          ///< the element type
    using view_type = ArraysView<T>;               ///< a view of the arrays
    using const_view_type = ArraysView<const T>;   ///< a read-only view of the arrays

    /// no arrays (e.g., to be assigned later)
    cuArrays() = default;
    /// count arrays of height x width, allocated in the device memory (not initialized)
    cuArrays(int height, int width, int count = 1);
    /// free the device memory
    ~cuArrays();
    /// the arrays of other, which is left empty
    cuArrays(cuArrays &&other) noexcept;
    /// free the device memory, and take the arrays of other, which is left empty
    cuArrays &operator=(cuArrays &&other) noexcept;
    cuArrays(const cuArrays &) = delete;
    cuArrays &operator=(const cuArrays &) = delete;

    /// the number of arrays
    int count() const { return data.count(); }
    /// the number of rows of an array (down, azimuth)
    int height() const { return data.height(); }
    /// the number of columns of an array (across, range)
    int width() const { return data.width(); }
    /// the number of elements of an array, height*width
    int arraySize() const { return data.arraySize(); }
    /// the number of elements, count*height*width
    std::size_t size() const { return data.size(); }
    /// whether there are no elements
    bool empty() const { return data.empty(); }
    /// the first element, in the device memory
    T *data_handle() { return data.data_handle(); }
    /// the first element, in the device memory (read-only)
    const T *data_handle() const { return data.data_handle(); }

    /// a view of the arrays, e.g., for the kernels
    view_type view() { return data; }
    /// a read-only view of the arrays
    const_view_type view() const { return data; }
    /// the arrays as a view
    operator view_type() { return data; }
    /// the arrays as a read-only view
    operator const_view_type() const { return data; }

    /// set the elements to zero (queued on the stream)
    void setZero(cudaStream_t stream);
    /// copy size() elements from the host memory (queued on the stream; host can be reused when
    /// this returns)
    void copyFromHost(const T *host, cudaStream_t stream);
    /// the elements, copied to the host memory once the work queued on the stream is done
    std::vector<T> toHost(cudaStream_t stream) const;
    /// write the arrays to the binary file fn (copied to the host memory as toHost)
    void outputToFile(const std::string &fn, cudaStream_t stream) const;

private:
    view_type data;   ///< the device memory, and the extents of the arrays
};

} // namespace
//end of file
