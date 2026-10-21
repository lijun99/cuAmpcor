/*
 * @file cuCorrNormalizer.h
 * @brief normalize the correlation surface (CPU backend)
 *
 * cuNormalizeProcessor is an abstract class for processors to normalize the correlation surface.
 * The CPU backend uses the sum area table based algorithm (cuNormalizeSAT) for all sizes.
 */

#pragma once

#include "forward.h"

namespace pycuampcor::cpu {

/**
 * Abstract class interface for correlation surface normalization processor
 * with different implementations
 */
class cuNormalizeProcessor {
public:
    // default constructor and destructor
    cuNormalizeProcessor() = default;
    virtual ~cuNormalizeProcessor() = default;
    // execute interface
    virtual void execute(cuArrays<real_type> * correlation, cuArrays<real_type> *reference, cuArrays<real_type> *secondary, stream_t stream) = 0;
};

// factory with the secondary dimension
std::unique_ptr<cuNormalizeProcessor> newCuNormalizer(int NX, int NY, int count);

class cuNormalizeSAT : public cuNormalizeProcessor
{
private:
    std::unique_ptr<cuArrays<double>> secondarySAT;
    std::unique_ptr<cuArrays<double>> secondarySAT2;

public:
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    ~cuNormalizeSAT();
    void execute(cuArrays<real_type> * correlation, cuArrays<real_type> *reference, cuArrays<real_type> *search, stream_t stream) override;
};

} // namespace
// end of file
