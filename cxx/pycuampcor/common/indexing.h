/**
 * @file  indexing.h
 * @brief Index helpers, for host and device code
 */

#pragma once

#include "hostDevice.h"

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
//end of file
