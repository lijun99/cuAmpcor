/**
 * @file  ArraysView.h
 * @brief A view of a batch of 2D arrays: the data of the arrays, without owning them
 *
 * The batches of windows, correlation surfaces, offsets, ... are count 2D arrays of
 * height x width, row-major, one after the other in memory. A view describes such a batch: a
 * pointer to its first element and its extents (count, height, width). It doesn't own the data:
 * it is copied (passed by value) to the kernels and workers, also to the CUDA kernels, as is.
 * The arrays owning the data (cpu::Arrays, cuda::cuArrays) convert to views.
 *
 * The interface follows std::mdspan (rank 3, row-major, extents (count, height, width)), with
 * what our code uses: extent(r), size(), data_handle(), and the element (k, i, j); plus the
 * height, width and count, the size of an array, the k-th array, and the elements by their
 * index in the batch.
 */

#pragma once

#include "hostDevice.h"

#include <cstddef>
#include <type_traits>

namespace pycuampcor {

/// a view of count 2D arrays of height x width of T (const T for read-only views), row-major,
/// contiguous
template <typename T>
class ArraysView {
public:
    using element_type = T;                       ///< the element type (const T for read-only views)
    using value_type = std::remove_cv_t<T>;       ///< the element type without const
    using index_type = int;                       ///< the type of the indices and extents
    using size_type = std::size_t;                ///< the type of the total number of elements
    using data_handle_type = T *;                 ///< the pointer to the elements
    using reference = T &;                        ///< the reference to an element

    /// the number of dimensions: count, height, width
    PYCUAMPCOR_HOST_DEVICE static constexpr int rank() { return 3; }

    /// an empty view
    ArraysView() = default;

    /// a view of count arrays of height x width, starting at data
    PYCUAMPCOR_HOST_DEVICE constexpr ArraysView(T *data, index_type height, index_type width,
        index_type count = 1)
        : data_(data), count_(count), height_(height), width_(width) {}

    /// a read-only view of a view (ArraysView<T> to ArraysView<const T>)
    template <typename U,
        typename = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    PYCUAMPCOR_HOST_DEVICE constexpr ArraysView(const ArraysView<U> &other)
        : ArraysView(other.data_handle(), other.height(), other.width(), other.count()) {}

    /// the extent of the dimension r: 0 the count, 1 the height, 2 the width
    PYCUAMPCOR_HOST_DEVICE constexpr index_type extent(int r) const
    {
        return r == 0 ? count_ : (r == 1 ? height_ : width_);
    }
    /// the number of arrays
    PYCUAMPCOR_HOST_DEVICE constexpr index_type count() const { return count_; }
    /// the number of rows of an array (down, azimuth)
    PYCUAMPCOR_HOST_DEVICE constexpr index_type height() const { return height_; }
    /// the number of columns of an array (across, range)
    PYCUAMPCOR_HOST_DEVICE constexpr index_type width() const { return width_; }
    /// the number of elements of an array, height*width
    PYCUAMPCOR_HOST_DEVICE constexpr index_type arraySize() const { return height_*width_; }
    /// the number of elements, count*height*width
    PYCUAMPCOR_HOST_DEVICE constexpr size_type size() const
    {
        return size_type(count_)*size_type(height_)*size_type(width_);
    }
    /// whether the view has no elements
    PYCUAMPCOR_HOST_DEVICE constexpr bool empty() const { return size() == 0; }
    /// the first element
    PYCUAMPCOR_HOST_DEVICE constexpr data_handle_type data_handle() const { return data_; }

    /// the element (i, j) of the k-th array
    PYCUAMPCOR_HOST_DEVICE constexpr reference operator()(index_type k, index_type i,
        index_type j) const
    {
        return data_[(k*height_ + i)*width_ + j];
    }
    /// the element (i, j) of the first array (e.g., of the only one)
    PYCUAMPCOR_HOST_DEVICE constexpr reference operator()(index_type i, index_type j) const
    {
        return data_[i*width_ + j];
    }
    /// the element idx in the batch (row-major over count, height, width)
    PYCUAMPCOR_HOST_DEVICE constexpr reference operator[](index_type idx) const
    {
        return data_[idx];
    }

    /// the k-th array, a view of one array
    PYCUAMPCOR_HOST_DEVICE constexpr ArraysView array(index_type k) const
    {
        return ArraysView(data_ + k*arraySize(), height_, width_, 1);
    }

private:
    T *data_ = nullptr;     ///< the first element
    index_type count_ = 0;  ///< the number of arrays
    index_type height_ = 0; ///< the number of rows of an array
    index_type width_ = 0;  ///< the number of columns of an array
};

} // namespace
//end of file
