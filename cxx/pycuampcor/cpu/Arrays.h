/**
 * @file  Arrays.h
 * @brief A batch of 2D arrays in the work memory (CPU)
 *
 * Arrays owns count arrays of height x width (row-major, one after the other) in the work (host)
 * memory, aligned for SIMD instructions: allocated when constructed, freed when destroyed. It can
 * be moved, not copied. The kernels and workers take views of the arrays (ArraysView.h), to which
 * the arrays convert. Its interface is that of cuda::cuArrays (the streams are not used).
 */

#pragma once

#include "forward.h"
#include "ArraysView.h"

namespace pycuampcor::cpu {

/// count 2D arrays of height x width of T in the work memory
template <typename T>
class Arrays {
public:
    using value_type = T;
    using view_type = ArraysView<T>;
    using const_view_type = ArraysView<const T>;

    /// no arrays (e.g., to be assigned later)
    Arrays() = default;
    /// count arrays of height x width, allocated in the work memory (not initialized)
    Arrays(int height, int width, int count = 1);
    /// free the memory
    ~Arrays();
    /// the arrays of other, which is left empty
    Arrays(Arrays &&other) noexcept;
    /// free the memory, and take the arrays of other, which is left empty
    Arrays &operator=(Arrays &&other) noexcept;
    Arrays(const Arrays &) = delete;
    Arrays &operator=(const Arrays &) = delete;

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
    /// the first element
    T *data_handle() { return data.data_handle(); }
    /// the first element (read-only)
    const T *data_handle() const { return data.data_handle(); }

    /// a view of the arrays, e.g., for the kernels
    view_type view() { return data; }
    /// a read-only view of the arrays
    const_view_type view() const { return data; }
    /// the arrays as a view
    operator view_type() { return data; }
    /// the arrays as a read-only view
    operator const_view_type() const { return data; }

    /// set the elements to zero
    void setZero(stream_t stream);
    /// copy size() elements from host
    void copyFromHost(const T *host, stream_t stream);
    /// a copy of the elements
    std::vector<T> toHost(stream_t stream) const;
    /// write the arrays to the binary file fn
    void outputToFile(const std::string &fn, stream_t stream) const;

private:
    view_type data;   ///< the work memory, and the extents of the arrays
};

} // namespace
//end of file
