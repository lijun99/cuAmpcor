/**
 * @file  ArraysFile.h
 * @brief Write a batch of 2D arrays to a binary file
 *
 * The output images and the intermediate arrays (debug dumps) are written as raw binary files:
 * the elements of the arrays, row-major, one array after the other.
 */

#pragma once

#include "ArraysView.h"
#include "logging.h"

#include <fstream>
#include <string>

namespace pycuampcor {

/// write the arrays (in the host memory) to the binary file fn
template <typename T>
void writeArraysToFile(const std::string &fn, ArraysView<const T> arrays)
{
    std::ofstream file(fn, std::ios_base::binary);
    file.write(reinterpret_cast<const char *>(arrays.data_handle()), arrays.size()*sizeof(T));
    logInfo(fn, " size: height ", arrays.height(), " width ", arrays.width(), " count ", arrays.count());
}

} // namespace
//end of file
