/*
 * @file  cuAmpcorProcessorTwoPass.h
 * @brief Ampcor processor for a batch of windows with TwoPass workflow
 *
 *
 */

#pragma once

#include "cuAmpcorProcessor.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/**
 * cuAmpcor processor for a chunk (a batch of windows)
 */
class cuAmpcorProcessorTwoPass : public cuAmpcorProcessor{
private:

    // local variables and workers

    // windows raw (not oversampled) data, complex and real
    std::unique_ptr<cuArrays<complex_type>> c_referenceBatchRaw;
    std::unique_ptr<cuArrays<complex_type>> c_secondaryBatchRaw;
    std::unique_ptr<cuArrays<complex_type>> c_secondaryBatchZoomIn;
    std::unique_ptr<cuArrays<real_type>> r_referenceBatchRaw;
    std::unique_ptr<cuArrays<real_type>> r_secondaryBatchRaw;

    // windows oversampled data
    std::unique_ptr<cuArrays<complex_type>> c_referenceBatchOverSampled;
    std::unique_ptr<cuArrays<complex_type>> c_secondaryBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_referenceBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_secondaryBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchRaw;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchZoomIn;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchZoomInOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchZoomInAdjust;

    // oversampling processors for complex images
    std::unique_ptr<cuOverSamplerC2C> referenceBatchOverSampler;
    std::unique_ptr<cuOverSamplerC2C> secondaryBatchOverSampler;

    // oversampling processor for correlation surface
    std::unique_ptr<cuOverSamplerR2R> corrOverSampler;
    std::unique_ptr<cuSincOverSamplerR2R> corrSincOverSampler;

    // cross-correlation processor with frequency domain algorithm
    std::unique_ptr<cuFreqCorrelator> cuCorrFreqDomain;
    std::unique_ptr<cuFreqCorrelator> cuCorrFreqDomain_OverSampled;

    // correlation surface normalizer
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerRaw;
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerOverSampled;

    // save offset results in different stages
    std::unique_ptr<cuArrays<int2>> offsetInit;
    std::unique_ptr<cuArrays<int2>> offsetZoomIn;
    std::unique_ptr<cuArrays<complex_type>> offsetFinal;
    std::unique_ptr<cuArrays<int2>> maxLocShift; // record the maxloc from the extract center
    std::unique_ptr<cuArrays<real_type>> corrMaxValue;
    std::unique_ptr<cuArrays<int2>> i_maxloc;
    std::unique_ptr<cuArrays<real_type>> r_maxval;

    // SNR estimation
    std::unique_ptr<cuArrays<real_type>> r_corrBatchRawZoomIn;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchSum;
    std::unique_ptr<cuArrays<int>> i_corrBatchZoomInValid;
    std::unique_ptr<cuArrays<int>> i_corrBatchValidCount;
    std::unique_ptr<cuArrays<real_type>> r_snrValue;

    // Variance estimation
    std::unique_ptr<cuArrays<real3_type>> r_covValue;

public:
    // constructor
    cuAmpcorProcessorTwoPass(cuAmpcorParameter *param_,
        cuArrays<complex_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        stream_t stream_);
    // destructor
    ~cuAmpcorProcessorTwoPass() override;

    // run the given chunk
    void run(int, int, const cuAmpcorChunk &) override;
};

} // namespace
