/**
 * @file  processors.cpp
 * @brief The processors of the chunks (the two-pass and one-pass workflows), for the arrays of
 *     the CPU package
 *
 * The processors are written once (common/cuAmpcorProcessor*Impl.h), as templates on the array
 * type; their kernels are found from the arrays of their arguments, those of this package.
 */

// the arrays of the package, and the types and workers that go with them
#include "Arrays.h"
#include "arrayTraits.h"
// the kernels
#include "ArraysCopy.h"
#include "CorrNormalization.h"
#include "CorrTimeDomain.h"
#include "Deramp.h"
#include "EstimateStats.h"
#include "Offset.h"
// the workers
#include "CorrFrequency.h"
#include "CorrNormalizer.h"
#include "OverSampler.h"
#include "SincOverSampler.h"
// the chunks loaded from the images
#include "cuAmpcorChunkLoader.h"
#include "float2.h"

// the factories of the workers, for the processors

/**
 * A normalizer for the correlation surfaces of count windows of height x width
 */
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cpu::Arrays>::normalizer_type>
pycuampcor::array_traits<pycuampcor::cpu::Arrays>::newNormalizer(int height, int width, int count)
{
    return cpu::newCuNormalizer(height, width, count);
}

/**
 * An oversampler (FFT or sinc, by method) of count correlation surfaces of height x width
 * around their peaks, by factor
 */
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cpu::Arrays>::surface_oversampler_type>
pycuampcor::array_traits<pycuampcor::cpu::Arrays>::newSurfaceOverSampler(OversamplingMethod method,
    int height, int width, int factor, int rawFactor, int count, stream_type stream)
{
    return cpu::newSurfaceOverSampler(method, height, width, factor, rawFactor, count, stream);
}

// the processors
#include "cuAmpcorProcessorImpl.h"
#include "cuAmpcorProcessorTwoPassImpl.h"
#include "cuAmpcorProcessorOnePassImpl.h"

// for the arrays of this package
template class pycuampcor::cuAmpcorProcessor<pycuampcor::cpu::Arrays>;
template class pycuampcor::cuAmpcorProcessorTwoPass<pycuampcor::cpu::Arrays>;
template class pycuampcor::cuAmpcorProcessorOnePass<pycuampcor::cpu::Arrays>;

//end of file
