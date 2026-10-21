/**
 * @file  externals.h
 * @brief The external headers used by the shared code (and by the cpu and cuda packages)
 */

#pragma once

// the standard library
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// hdf5, to read images from HDF5 datasets
#ifdef PYCUAMPCOR_WITH_HDF5
#include <hdf5.h>
#endif

//end of file
