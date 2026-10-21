/**
 * @file  public.h
 * @brief The CUDA package: everything, in dependency order
 *
 * The package (namespace pycuampcor::cuda) is the library pycuampcor_cuda, with the shared code
 * (namespace pycuampcor, library pycuampcor_common); clients include this header, with the
 * include directories of the library (cuda/, then common/).
 */

#pragma once

// external packages
#include "externals.h"
// forward declarations
#include "forward.h"
// published type aliases
#include "api.h"

// the shared code: parameters, options, logging, images, layers
#include "../common/public.h"

// the vector types and their operators
#include "data_types.h"
#include "float2.h"
// utilities
#include "cudaError.h"
#include "device.h"
#include "cufftPlan.h"

// batches of 2D arrays
#include "cuArrays.h"
// kernels on batches of arrays
#include "cuArraysCopy.h"
#include "cuArraysPadding.h"
#include "cuDeramp.h"
#include "cuCorrNormalization.h"
#include "cuCorrNormalizationSAT.h"
#include "cuCorrTimeDomain.h"
#include "cuOffset.h"
#include "cuEstimateStats.h"
// workers with their own buffers and plans
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"

// the processing: chunks loaded from the images, processors of the chunks, and the controller
#include "cuChunkLoader.h"
#include "traits.h"
#include "Processor.h"
#include "ProcessorTwoPass.h"
#include "ProcessorOnePass.h"
#include "cuController.h"

//end of file
