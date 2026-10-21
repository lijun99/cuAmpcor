/**
 * @file  backend_public.h
 * @brief The CPU backend, in dependency order (see public.h)
 */

#pragma once

// the backend hooks of the shared code
#include "backend.h"
// the vector types and their operators
#include "data_types.h"
#include "float2.h"
// utilities
#include "fftwUtil.h"

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

//end of file
