/**
 * @file  public.h
 * @brief The CPU package: everything, in dependency order
 *
 * The package (namespace pycuampcor::cpu) is the library pycuampcor_cpu, with the shared code
 * (namespace pycuampcor, library pycuampcor_common); clients include this header, with the
 * include directories of the library (cpu/, then common/).
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
#include "fftwUtil.h"

// batches of 2D arrays
#include "Arrays.h"
// kernels on batches of arrays
#include "ArraysCopy.h"
#include "ArraysPadding.h"
#include "Deramp.h"
#include "CorrNormalization.h"
#include "CorrNormalizationSAT.h"
#include "CorrTimeDomain.h"
#include "Offset.h"
#include "EstimateStats.h"
// workers with their own buffers and plans
#include "FFTOverSampler.h"
#include "SincOverSampler.h"
#include "CorrFrequency.h"
#include "CorrNormalizer.h"

// the processing: chunks loaded from the images, processors of the chunks, and the controller
#include "ChunkLoader.h"
#include "traits.h"
#include "Processor.h"
#include "ProcessorTwoPass.h"
#include "ProcessorOnePass.h"
#include "Controller.h"

//end of file
