/*
 * @file  cuAmpcorProcessorOnePass.h
 * @brief Ampcor processor for a batch of windows with OnePass workflow
 *
 *
 */

#ifndef __CUAMPCORPROCESSOROnePass_H
#define __CUAMPCORPROCESSOROnePass_H

#include "cuAmpcorProcessor.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {


/**
 * cuAmpcor processor for a chunk (a batch of windows)
 */
class cuAmpcorProcessorOnePass : public cuAmpcorProcessor {

private:

    // local variables and workers

    // windows raw (not oversampled) data, complex and real
    std::unique_ptr<cuArrays<complex_type>> c_referenceBatchRaw;
    std::unique_ptr<cuArrays<complex_type>> c_secondaryBatchRaw;
    // cuArrays<real_type> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // windows oversampled data
    std::unique_ptr<cuArrays<complex_type>> c_referenceBatchOverSampled;
    std::unique_ptr<cuArrays<complex_type>> c_secondaryBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_referenceBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_secondaryBatchOverSampled;
    std::unique_ptr<cuArrays<real_type>> r_corrBatch;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchZoomIn;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchZoomInOverSampled;

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
    int2 maxLocShift; // record the maxloc from the extract center
    std::unique_ptr<cuArrays<real_type>> corrMaxValue;
    std::unique_ptr<cuArrays<int2>> i_maxloc;
    std::unique_ptr<cuArrays<real_type>> r_maxval;

    // SNR estimation, as in the two-pass workflow: the correlation surface around the peak
    // at the raw pixel spacing, its valid pixels and their count
    std::unique_ptr<cuArrays<real_type>> r_corrBatchRawZoomIn;
    std::unique_ptr<cuArrays<int>> i_corrBatchZoomInValid;
    std::unique_ptr<cuArrays<int>> i_corrBatchValidCount;
    std::unique_ptr<cuArrays<real_type>> r_corrBatchSum;
    std::unique_ptr<cuArrays<real_type>> r_snrValue;

    // Variance estimation
    std::unique_ptr<cuArrays<real3_type>> r_covValue;

public:
    // constructor
    cuAmpcorProcessorOnePass(cuAmpcorParameter *param_,
        cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_,
        cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
        stream_t stream_);
    // destructor
    ~cuAmpcorProcessorOnePass() override;

    // run the given chunk
    void run(int, int, const cuAmpcorChunk &) override;
};

} // namespace

#endif //__CUAMPCORPROCESSOROnePass_H
