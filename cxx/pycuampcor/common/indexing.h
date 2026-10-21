/**
 * @file  indexing.h
 * @brief Index helpers, for host and device code
 */

#ifndef __PYCUAMPCOR_INDEXING_H
#define __PYCUAMPCOR_INDEXING_H

// functions callable from host and device (CUDA) code
#ifdef __CUDACC__
#define PYCUAMPCOR_HOST_DEVICE __host__ __device__
#else
#define PYCUAMPCOR_HOST_DEVICE
#endif

namespace pycuampcor {

/// index of the element (i, j) of a row-major 2D array with nj columns
PYCUAMPCOR_HOST_DEVICE constexpr int idx2r(int i, int j, int nj)
{
    return i*nj + j;
}

/// i/j rounded up (i >= 0, j > 0), e.g., the number of blocks of j threads for i elements
PYCUAMPCOR_HOST_DEVICE constexpr int divUp(int i, int j)
{
    return (i + j - 1)/j;
}

} // namespace

#endif //__PYCUAMPCOR_INDEXING_H
//end of file
