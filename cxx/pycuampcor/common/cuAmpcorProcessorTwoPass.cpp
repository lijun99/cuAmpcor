#include "cuAmpcorProcessorTwoPass.h"
#include "logging.h"
#include "cuArraysCopy.h"
#include "cuDeramp.h"
#include "cuCorrNormalization.h"
#include "cuOffset.h"
#include "cuCorrTimeDomain.h"
#include "cuEstimateStats.h"
#include "debug.h"
#include "cuAmpcorChunkLoader.h"
#include "cuAmpcorParameter.h"
#include "cuCorrFrequency.h"
#include "cuCorrNormalizer.h"
#include "cuOverSampler.h"
#include "cuSincOverSampler.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/**
 * Run ampcor process for a batch of images (a chunk)
 * @param[in] idxDown_  index of the chunk along Down/Azimuth direction
 * @param[in] idxAcross_ index of the chunk along Across/Range direction
 */
void cuAmpcorProcessorTwoPass::run(int idxDown_, int idxAcross_, const cuAmpcorChunk &chunk)
{
    // set chunk index
    setIndex(idxDown_, idxAcross_);

    // copy the reference windows from the loaded chunk
    copyToBatch(chunk.reference, param->referenceStartPixelDown, param->referenceStartPixelAcross, c_referenceBatchRaw.get());
    // take amplitudes
    cuArraysAbs(c_referenceBatchRaw.get(), r_referenceBatchRaw.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRaw.get()", stream);
    r_referenceBatchRaw->outputToFile("r_referenceBatchRaw.get()", stream);
#endif

    // compute and subtract mean values (for normalized)
    cuArraysSubtractMean(r_referenceBatchRaw.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    r_referenceBatchRaw->outputToFile("r_referenceBatchRawSubMean", stream);
#endif

    // copy the secondary windows from the loaded chunk
    copyToBatch(chunk.secondary, param->secondaryStartPixelDown, param->secondaryStartPixelAcross, c_secondaryBatchRaw.get());
    // take amplitudes
    cuArraysAbs(c_secondaryBatchRaw.get(), r_secondaryBatchRaw.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the raw secondary image(s)
    c_secondaryBatchRaw->outputToFile("c_secondaryBatchRaw.get()", stream);
    r_secondaryBatchRaw->outputToFile("r_secondaryBatchRaw.get()", stream);
#endif

    //cross correlation for un-oversampled data
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain->execute(r_referenceBatchRaw.get(), r_secondaryBatchRaw.get(), r_corrBatchRaw.get());
    } else {
        cuCorrTimeDomain(r_referenceBatchRaw.get(), r_secondaryBatchRaw.get(), r_corrBatchRaw.get(), stream); //time domain cross correlation
    }

#ifdef CUAMPCOR_DEBUG
    // dump the un-normalized correlation surface
    r_corrBatchRaw->outputToFile("r_corrBatchRawUnNorm", stream);
#endif

    // normalize the correlation surface
    corrNormalizerRaw->execute(r_corrBatchRaw.get(), r_referenceBatchRaw.get(), r_secondaryBatchRaw.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the normalized correlation surface
    r_corrBatchRaw->outputToFile("r_corrBatchRaw.get()", stream);
#endif

    // find the maximum location of none-oversampled correlation
    // 41 x 41, if halfsearchrange=20
    cuArraysMaxloc2D(r_corrBatchRaw.get(), offsetInit.get(), r_maxval.get(), stream);

    // estimate variance
    cuEstimateVariance(r_corrBatchRaw.get(), offsetInit.get(), r_maxval.get(), r_referenceBatchRaw->size, 1, r_covValue.get(), stream);

    // estimate SNR
    // step1: extraction of correlation surface around the peak
    cuArraysCopyExtractCorr(r_corrBatchRaw.get(), r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(), offsetInit.get(), stream);

    // step2: summation of correlation and data point values
    cuArraysSumCorr(r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(), r_corrBatchSum.get(), i_corrBatchValidCount.get(), stream);

#ifdef CUAMPCOR_DEBUG
    r_maxval->outputToFile("r_maxval.get()", stream);
    r_corrBatchRawZoomIn->outputToFile("r_corrBatchRawStatZoomIn", stream);
    i_corrBatchZoomInValid->outputToFile("i_corrBatchZoomInValid.get()", stream);
    r_corrBatchSum->outputToFile("r_corrBatchSum.get()", stream);
    i_corrBatchValidCount->outputToFile("i_corrBatchValidCount.get()", stream);
#endif

    // step3: divide the peak value by the mean of surrounding values
    cuEstimateSnr(r_corrBatchSum.get(), i_corrBatchValidCount.get(), r_maxval.get(), r_snrValue.get(), stream);

#ifdef CUAMPCOR_DEBUG
    offsetInit->outputToFile("i_offsetInit", stream);
    r_snrValue->outputToFile("r_snrValue.get()", stream);
    r_covValue->outputToFile("r_covValue.get()", stream);
#endif

    // Using the approximate estimation to adjust secondary image (half search window size becomes only 4 pixels)
    // determine the starting pixel to extract secondary images around the max location
    cuDetermineSecondaryExtractOffset(offsetInit.get(),
        maxLocShift.get(),
        param->halfSearchRangeDownRaw, // old range
        param->halfSearchRangeAcrossRaw,
        param->halfZoomWindowSizeRaw,  // new range
        param->halfZoomWindowSizeRaw,
        stream);

#ifdef CUAMPCOR_DEBUG
    offsetInit->outputToFile("i_offsetInitAdjusted", stream);
    maxLocShift->outputToFile("i_maxLocShift", stream);
#endif

    // deramp reference
    cuDeramp(param->derampMethod, c_referenceBatchRaw.get(), param->derampAxis, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the deramped reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRawDeramped", stream);
#endif

    // oversample reference
    referenceBatchOverSampler->execute(c_referenceBatchRaw.get(), c_referenceBatchOverSampled.get());

    // take amplitudes
    cuArraysAbs(c_referenceBatchOverSampled.get(), r_referenceBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled reference image(s)
    c_referenceBatchOverSampled->outputToFile("c_referenceBatchOverSampled.get()", stream);
    r_referenceBatchOverSampled->outputToFile("r_referenceBatchOverSampled.get()", stream);
#endif

    // compute and subtract the mean value
    cuArraysSubtractMean(r_referenceBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled reference image(s) with mean subtracted
    r_referenceBatchOverSampled->outputToFile("r_referenceBatchOverSampledSubMean",stream);
#endif

    // extract secondary
    cuArraysCopyExtract(c_secondaryBatchRaw.get(), c_secondaryBatchZoomIn.get(), offsetInit.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the extracted raw secondary image
    c_secondaryBatchZoomIn->outputToFile("c_secondaryBatchZoomInRaw", stream);
#endif
    // deramp secondary
    cuDeramp(param->derampMethod, c_secondaryBatchZoomIn.get(), param->derampAxis, stream);
#ifdef CUAMPCOR_DEBUG

    // dump the deramped secondary image(s)
    c_secondaryBatchZoomIn->outputToFile("c_secondaryBatchZoomInDeramped", stream);
#endif

    // oversample secondary
    secondaryBatchOverSampler->execute(c_secondaryBatchZoomIn.get(), c_secondaryBatchOverSampled.get());
    // take amplitudes
    cuArraysAbs(c_secondaryBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the extracted raw secondary image
    c_secondaryBatchZoomIn->outputToFile("c_secondaryBatchZoomIn.get()", stream);
    // dump the oversampled secondary image(s)
    c_secondaryBatchOverSampled->outputToFile("c_secondaryBatchOverSampled.get()", stream);
    r_secondaryBatchOverSampled->outputToFile("r_secondaryBatchOverSampled.get()", stream);
#endif

    // correlate oversampled images
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain_OverSampled->execute(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatchZoomIn.get());
    }
    else {
        cuCorrTimeDomain(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatchZoomIn.get(), stream);
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (un-normalized)
    r_corrBatchZoomIn->outputToFile("r_corrBatchZoomInUnNorm", stream);
#endif

    // normalize the correlation surface
    corrNormalizerOverSampled->execute(r_corrBatchZoomIn.get(), r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (normalized)
    r_corrBatchZoomIn->outputToFile("r_corrBatchZoomIn.get()", stream);
#endif

    // remove the last row and col to get even sequences
    cuArraysCopyExtract(r_corrBatchZoomIn.get(), r_corrBatchZoomInAdjust.get(), make_int2(0,0), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the adjusted correlation Surface
    r_corrBatchZoomInAdjust->outputToFile("r_corrBatchZoomInAdjust.get()", stream);
#endif

    // oversample the correlation surface
    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        // sinc interpolator only computes (-i_sincwindow, i_sincwindow)*oversampling factor
        // we need the max loc as the center if shifted
        corrSincOverSampler->execute(r_corrBatchZoomInAdjust.get(), r_corrBatchZoomInOverSampled.get(),
            maxLocShift.get(), param->corrSurfaceOverSamplingFactor*param->rawDataOversamplingFactor
            );
    }
    else {
        corrOverSampler->execute(r_corrBatchZoomInAdjust.get(), r_corrBatchZoomInOverSampled.get());
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface
    r_corrBatchZoomInOverSampled->outputToFile("r_corrBatchZoomInOverSampled.get()", stream);
#endif

    //find the max again
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled.get(), offsetZoomIn.get(), corrMaxValue.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the max location on oversampled correlation surface
    offsetZoomIn->outputToFile("i_offsetZoomIn", stream);
    corrMaxValue->outputToFile("r_maxvalZoomInOversampled", stream);
#endif

    // determine the final offset from non-oversampled (pixel) and oversampled (sub-pixel)
    // = (Init-HalfsearchRange) + ZoomIn/(2*ovs)
    cuSubPixelOffset2Pass(offsetInit.get(), offsetZoomIn.get(), offsetFinal.get(),
        param->corrSurfaceOverSamplingFactor, param->rawDataOversamplingFactor,
        param->halfSearchRangeDownRaw, param->halfSearchRangeAcrossRaw,
        stream);

    // Insert the chunk results to final images
    cuArraysCopyInsert(offsetFinal.get(), offsetImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // snr
    cuArraysCopyInsert(r_snrValue.get(), snrImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // Variance.
    cuArraysCopyInsert(r_covValue.get(), covImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // peak value.
    cuArraysCopyInsert(corrMaxValue.get(), peakValueImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // all done

}

/// constructor
cuAmpcorProcessorTwoPass::cuAmpcorProcessorTwoPass(cuAmpcorParameter *param_,
    cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_, cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
    stream_t stream_)
    : cuAmpcorProcessor(param_, offsetImage_, snrImage_, covImage_, peakValueImage_, stream_)
{
    param = param_;
    offsetImage = offsetImage_;
    snrImage = snrImage_;
    covImage = covImage_;

    stream = stream_;

    c_referenceBatchRaw = std::make_unique<cuArrays<complex_type>>(
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_referenceBatchRaw->allocate();

    c_secondaryBatchRaw = std::make_unique<cuArrays<complex_type>>(
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchRaw->allocate();

    r_referenceBatchRaw = std::make_unique<cuArrays<real_type>>(
        param->windowSizeHeightRaw, param->windowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_referenceBatchRaw->allocate();

    r_secondaryBatchRaw = std::make_unique<cuArrays<real_type>>(
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_secondaryBatchRaw->allocate();

    c_secondaryBatchZoomIn = std::make_unique<cuArrays<complex_type>>(
        param->searchWindowSizeHeightRawZoomIn, param->searchWindowSizeWidthRawZoomIn,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchZoomIn->allocate();

    c_referenceBatchOverSampled = std::make_unique<cuArrays<complex_type>>(
            param->windowSizeHeight, param->windowSizeWidth,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_referenceBatchOverSampled->allocate();

    c_secondaryBatchOverSampled = std::make_unique<cuArrays<complex_type>>(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchOverSampled->allocate();

    r_referenceBatchOverSampled = std::make_unique<cuArrays<real_type>>(
            param->windowSizeHeight, param->windowSizeWidth,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_referenceBatchOverSampled->allocate();

    r_secondaryBatchOverSampled = std::make_unique<cuArrays<real_type>>(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_secondaryBatchOverSampled->allocate();

    referenceBatchOverSampler = std::make_unique<cuOverSamplerC2C>(
        c_referenceBatchRaw->height, c_referenceBatchRaw->width, //original size
        c_referenceBatchOverSampled->height, c_referenceBatchOverSampled->width, //oversampled size
        c_referenceBatchRaw->count, stream);

    secondaryBatchOverSampler = std::make_unique<cuOverSamplerC2C>(c_secondaryBatchZoomIn->height, c_secondaryBatchZoomIn->width,
            c_secondaryBatchOverSampled->height, c_secondaryBatchOverSampled->width, c_secondaryBatchRaw->count, stream);

    r_corrBatchRaw = std::make_unique<cuArrays<real_type>>(
            param->searchWindowSizeHeightRaw-param->windowSizeHeightRaw+1,
            param->searchWindowSizeWidthRaw-param->windowSizeWidthRaw+1,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchRaw->allocate();

    r_corrBatchZoomIn = std::make_unique<cuArrays<real_type>>(
            param->searchWindowSizeHeight - param->windowSizeHeight+1,
            param->searchWindowSizeWidth - param->windowSizeWidth+1,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchZoomIn->allocate();

    r_corrBatchZoomInAdjust = std::make_unique<cuArrays<real_type>>(
            param->searchWindowSizeHeight - param->windowSizeHeight,
            param->searchWindowSizeWidth - param->windowSizeWidth,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchZoomInAdjust->allocate();

    r_corrBatchZoomInOverSampled = std::make_unique<cuArrays<real_type>>(
        param->corrZoomWindowSize * param->corrSurfaceOverSamplingFactor,
        param->corrZoomWindowSize * param->corrSurfaceOverSamplingFactor,
        param->windowsDownInChunk,
        param->windowsAcrossInChunk);
    r_corrBatchZoomInOverSampled->allocate();

    offsetInit = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetInit->allocate();

    offsetZoomIn = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetZoomIn->allocate();

    offsetFinal = std::make_unique<cuArrays<real2_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetFinal->allocate();

    maxLocShift = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    maxLocShift->allocate();

    corrMaxValue = std::make_unique<cuArrays<real_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    corrMaxValue->allocate();

    // new arrays due to snr estimation
    r_corrBatchRawZoomIn = std::make_unique<cuArrays<real_type>>(
            param->corrRawZoomInHeight,
            param->corrRawZoomInWidth,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchRawZoomIn->allocate();

    i_corrBatchZoomInValid = std::make_unique<cuArrays<int>>(
            param->corrRawZoomInHeight,
            param->corrRawZoomInWidth,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    i_corrBatchZoomInValid->allocate();

    r_corrBatchSum = std::make_unique<cuArrays<real_type>>(
                    param->windowsDownInChunk,
                    param->windowsAcrossInChunk);
    r_corrBatchSum->allocate();

    i_corrBatchValidCount = std::make_unique<cuArrays<int>>(
                        param->windowsDownInChunk,
                        param->windowsAcrossInChunk);
    i_corrBatchValidCount->allocate();

    i_maxloc = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);

    i_maxloc->allocate();

    r_maxval = std::make_unique<cuArrays<real_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);

    r_maxval->allocate();

    r_snrValue = std::make_unique<cuArrays<real_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);

    r_snrValue->allocate();

    r_covValue = std::make_unique<cuArrays<real3_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);

    r_covValue->allocate();

    // end of new arrays

    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        corrSincOverSampler = std::make_unique<cuSincOverSamplerR2R>(param->corrSurfaceOverSamplingFactor, stream);
    }
    else {
        corrOverSampler = std::make_unique<cuOverSamplerR2R>(param->corrZoomWindowSize, param->corrZoomWindowSize,
            (param->corrZoomWindowSize)*param->corrSurfaceOverSamplingFactor,
            (param->corrZoomWindowSize)*param->corrSurfaceOverSamplingFactor,
            param->windowsDownInChunk*param->windowsAcrossInChunk,
            stream);
    }
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain = std::make_unique<cuFreqCorrelator>(
            param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
            param->windowsDownInChunk*param->windowsAcrossInChunk,
            stream);
        cuCorrFreqDomain_OverSampled = std::make_unique<cuFreqCorrelator>(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk * param->windowsAcrossInChunk,
            stream);
    }

    corrNormalizerRaw = newCuNormalizer(
        param->searchWindowSizeHeightRaw,
        param->searchWindowSizeWidthRaw,
        param->windowsDownInChunk * param->windowsAcrossInChunk
        );

    corrNormalizerOverSampled =
        newCuNormalizer(
        param->searchWindowSizeHeight,
        param->searchWindowSizeWidth,
        param->windowsDownInChunk * param->windowsAcrossInChunk
        );

    logDebug("all objects in chunk are created");
}

// destructor
cuAmpcorProcessorTwoPass::~cuAmpcorProcessorTwoPass() = default;

// end of file

} // namespace