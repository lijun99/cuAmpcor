/**
 * @file  public.h
 * @brief The package: everything, in dependency order
 *
 * The package is compiled for each backend, with the include directories common/ and the
 * backend's (cpu/ or cuda/), and PYCUAMPCOR_BACKEND set to the backend; clients include this
 * header (with the include directories of the library pycuampcor_<backend>).
 */

#pragma once

// external packages
#include "externals.h"
// forward declarations
#include "forward.h"
// published type aliases
#include "api.h"

// support
#include "debug.h"
#include "hostDevice.h"
#include "indexing.h"
#include "logging.h"
#include "cuAmpcorOptions.h"

// the backend: arrays, kernels, fft, oversamplers, correlators, normalizers
#include "backend_public.h"

// the processing
#include "cuAmpcorParameter.h"
#include "SlcImage.h"
#ifdef PYCUAMPCOR_WITH_HDF5
#include "Hdf5SlcImage.h"
#endif
#include "cuAmpcorChunkLoader.h"
#include "cuAmpcorProcessor.h"
#include "cuAmpcorProcessorTwoPass.h"
#include "cuAmpcorProcessorOnePass.h"
#include "cuAmpcorController.h"

//end of file
