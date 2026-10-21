/**
 * @file  processors.cpp
 * @brief The processors of the chunks (the two-pass and one-pass workflows), for the arrays of
 *     the CPU package
 *
 * The processors are written once (common/Processor*Impl.h), as templates on the array
 * type; their kernels are found from the arrays of their arguments, those of this package.
 */

// the arrays of the package, and the types and workers that go with them
#include "Arrays.h"
#include "traits.h"
// the kernels
#include "ArraysCopy.h"
#include "CorrNormalization.h"
#include "CorrTimeDomain.h"
#include "Deramp.h"
#include "EstimateStats.h"
#include "Offset.h"
// the workers
#include "CorrCoherence.h"
#include "CorrFrequency.h"
#include "CorrNormalizer.h"
#include "FFTOverSampler.h"
#include "SincOverSampler.h"
// the chunks loaded from the images
#include "ChunkLoader.h"
#include "float2.h"

// the factories of the workers, for the processors

/**
 * A normalizer for the correlation surfaces of count windows of height x width
 */
std::unique_ptr<pycuampcor::array_traits<pycuampcor::cpu::Arrays>::normalizer_type>
pycuampcor::array_traits<pycuampcor::cpu::Arrays>::newNormalizer(int height, int width, int count)
{
    return cpu::newNormalizer(height, width, count);
}

// the processors
#include "ProcessorImpl.h"
#include "ProcessorTwoPassImpl.h"
#include "ProcessorOnePassImpl.h"

// for the arrays of this package
template class pycuampcor::Processor<pycuampcor::cpu::Arrays>;
template class pycuampcor::ProcessorTwoPass<pycuampcor::cpu::Arrays>;
template class pycuampcor::ProcessorOnePass<pycuampcor::cpu::Arrays>;

//end of file
