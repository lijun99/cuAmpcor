/**
 * @file  public.h
 * @brief The shared code: everything, in dependency order
 *
 * The shared code (namespace pycuampcor) is compiled once, into the library pycuampcor_common;
 * the cpu and cuda packages include it from their own public.h.
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

// the parameters
#include "cuAmpcorParameter.h"
// the images
#include "SlcImage.h"
#ifdef PYCUAMPCOR_WITH_HDF5
#include "Hdf5SlcImage.h"
#endif
// layers processed together
#include "cuAmpcorLayers.h"
// the processors on the arrays of a package: what goes with the arrays, the interfaces of the
// workers, and the workflows (templates, instantiated by the packages)
#include "cuAmpcorArrayTraits.h"
#include "cuNormalizer.h"
#include "cuSurfaceOverSampler.h"

//end of file
