/**
 * @file  processors.cpp
 * @brief The processors of the chunks (the two-pass and one-pass workflows), for the arrays of
 *     the CUDA package
 *
 * The processors are written once (common/Processor*Impl.h), as templates on the array
 * type; their kernels are found from the arrays of their arguments, those of this package.
 */

// the arrays of the package, and the types and workers that go with them
#include "cuArrays.h"
#include "traits.h"
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
#include "cuFFTOverSampler.h"
#include "cuSincOverSampler.h"
// the chunks loaded from the images
#include "cuChunkLoader.h"
#include "float2.h"

// the factories of the workers, for the processors

/**
 * A normalizer for the correlation surfaces of count windows of height x width
 */
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::normalizer_type>
pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::newNormalizer(int height, int width, int count)
{
    return cuda::newNormalizer(height, width, count);
}

/**
 * An oversampler (FFT or sinc, by method) of count correlation surfaces of height x width
 * around their peaks, by factor
 */
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::surface_oversampler_type>
pycuampcor::array_traits<pycuampcor::cuda::cuArrays>::newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, stream_type stream)
{
    return cuda::newSurfaceOverSampler(method, height, width, factor, rawFactor, count, stream);
}

// the processors
#include "ProcessorImpl.h"
#include "ProcessorTwoPassImpl.h"
#include "ProcessorOnePassImpl.h"

// for the arrays of this package
template class pycuampcor::Processor<pycuampcor::cuda::cuArrays>;
template class pycuampcor::ProcessorTwoPass<pycuampcor::cuda::cuArrays>;
template class pycuampcor::ProcessorOnePass<pycuampcor::cuda::cuArrays>;

//end of file
