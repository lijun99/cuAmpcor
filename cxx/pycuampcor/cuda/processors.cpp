/**
 * @file  processors.cpp
 * @brief The processors of the chunks (the two-pass and one-pass workflows), for the arrays of
 *     the CUDA package
 *
 * The processors are written once (common/cuAmpcorProcessor*.icc), as templates on the array
 * type; their kernels are found from the arrays of their arguments, those of this package.
 */

// the arrays of the package, and the types and workers that go with them
#include "cuArrays.h"
#include "arrayTraits.h"
// the kernels
#include "cuArraysCopy.h"
#include "cuCorrNormalization.h"
#include "cuCorrTimeDomain.h"
#include "cuDeramp.h"
#include "cuEstimateStats.h"
#include "cuOffset.h"
// the workers
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"
// the chunks loaded from the images
#include "cuAmpcorChunkLoader.h"
#include "float2.h"

// the factories of the workers, for the processors
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::normalizer_type>
pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::newNormalizer(int height, int width, int count)
{
    return cuda::newCuNormalizer(height, width, count);
}

std::unique_ptr<pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::surface_oversampler_type>
pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, stream_type stream)
{
    return cuda::newSurfaceOverSampler(method, height, width, factor, rawFactor, count, stream);
}

// the processors
#include "cuAmpcorProcessor.icc"
#include "cuAmpcorProcessorTwoPass.icc"
#include "cuAmpcorProcessorOnePass.icc"

// for the arrays of this package
template class pycuampcor::cuAmpcorProcessor<pycuampcor::cuda::cuArrays>;
template class pycuampcor::cuAmpcorProcessorTwoPass<pycuampcor::cuda::cuArrays>;
template class pycuampcor::cuAmpcorProcessorOnePass<pycuampcor::cuda::cuArrays>;

//end of file
