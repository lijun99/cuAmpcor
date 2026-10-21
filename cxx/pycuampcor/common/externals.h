/**
 * @file  externals.h
 * @brief The external headers used by the package, and the backend's implementation choices
 */

#pragma once

// the standard library
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// hdf5, to read images from HDF5 datasets
#ifdef PYCUAMPCOR_WITH_HDF5
#include <hdf5.h>
#endif

// the backend: the precision and vector types, and its libraries (fftw or cuda/cufft)
#include "backend_externals.h"

//end of file
