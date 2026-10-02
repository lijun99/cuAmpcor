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
    cuArrays<complex_type> * c_referenceBatchRaw, * c_secondaryBatchRaw;
    // cuArrays<real_type> * r_referenceBatchRaw, * r_secondaryBatchRaw;

    // windows oversampled data
    cuArrays<complex_type> * c_referenceBatchOverSampled, * c_secondaryBatchOverSampled;
    cuArrays<real_type> * r_referenceBatchOverSampled, * r_secondaryBatchOverSampled;
    cuArrays<real_type> * r_corrBatch, * r_corrBatchZoomIn, * r_corrBatchZoomInOverSampled;

    // oversampling processors for complex images
    cuOverSamplerC2C *referenceBatchOverSampler, *secondaryBatchOverSampler;

    // oversampling processor for correlation surface
    cuOverSamplerR2R *corrOverSampler;
    cuSincOverSamplerR2R *corrSincOverSampler;

    // cross-correlation processor with frequency domain algorithm
    cuFreqCorrelator *cuCorrFreqDomain, *cuCorrFreqDomain_OverSampled;

    // correlation surface normalizer
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerRaw;
    std::unique_ptr<cuNormalizeProcessor> corrNormalizerOverSampled;

    // save offset results in different stages
    cuArrays<int2> *offsetInit;
    cuArrays<int2> *offsetZoomIn;
    cuArrays<complex_type> *offsetFinal;
    int2 maxLocShift; // record the maxloc from the extract center
    cuArrays<real_type> *corrMaxValue;
    cuArrays<int2> *i_maxloc;
    cuArrays<real_type> *r_maxval;

    // SNR estimation, as in the two-pass workflow: the correlation surface around the peak
    // at the raw pixel spacing, its valid pixels and their count
    cuArrays<real_type> *r_corrBatchRawZoomIn;
    cuArrays<int> *i_corrBatchZoomInValid, *i_corrBatchValidCount;
    cuArrays<real_type> *r_corrBatchSum;
    cuArrays<real_type> *r_snrValue;

    // Variance estimation
    cuArrays<real3_type> *r_covValue;

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
