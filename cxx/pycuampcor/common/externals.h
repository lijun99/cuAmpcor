/**
 * @file  externals.h
 * @brief The external headers used by the package, and the backend's implementation choices
 */

#pragma once

// the standard library
#include <algorithm>
#include <atomic>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// posix: files and memory maps
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// hdf5 and zlib, to read images from HDF5 datasets (and decode their chunks)
#ifdef PYCUAMPCOR_WITH_HDF5
#include <hdf5.h>
#include <zlib.h>
#endif

// the backend: the precision and vector types, and its libraries (fftw or cuda/cufft)
#include "backend_externals.h"

//end of file
