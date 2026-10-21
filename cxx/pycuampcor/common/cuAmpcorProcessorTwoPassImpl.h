/**
 * @file  cuAmpcorProcessorTwoPassImpl.h
 * @brief The two-pass workflow: the implementation
 *
 * The definitions of the member functions of cuAmpcorProcessorTwoPass (templates); only the packages'
 * processors.cpp include them (see cuAmpcorProcessorImpl.h).
 */

#pragma once

#include "cuAmpcorProcessorTwoPass.h"
#include "cuAmpcorOptions.h"
#include "cuAmpcorParameter.h"
#include "cuNormalizer.h"
#include "logging.h"

namespace pycuampcor {

/**
 * Process the chunk (idxDown, idxAcross), and write its results to the output images
 * @param idxDown, idxAcross the index of the chunk, down and across
 * @param chunk the chunk, loaded by the chunk loader
 */
template <template <typename> class arrayT>
void cuAmpcorProcessorTwoPass<arrayT>::run(int idxDown, int idxAcross, const chunk_type &chunk)
{
    using traits = typename base_type::traits;
    setIndex(idxDown, idxAcross);

    // ==== first pass, at the raw pixel spacing ====

    // the reference windows: their amplitudes, with the mean subtracted
    copyToBatch(chunk.reference, param.referenceStartPixelDown, param.referenceStartPixelAcross,
        c_referenceBatchRaw.get());
    cuArraysAbs(*c_referenceBatchRaw, *r_referenceBatchRaw, stream);
    dump(c_referenceBatchRaw.get(), "c_referenceBatchRaw");
    dump(r_referenceBatchRaw.get(), "r_referenceBatchRaw");
    cuArraysSubtractMean(*r_referenceBatchRaw, stream);
    dump(r_referenceBatchRaw.get(), "r_referenceBatchRawSubMean");

    // the secondary windows: their amplitudes
    copyToBatch(chunk.secondary, param.secondaryStartPixelDown, param.secondaryStartPixelAcross,
        c_secondaryBatchRaw.get());
    cuArraysAbs(*c_secondaryBatchRaw, *r_secondaryBatchRaw, stream);
    dump(c_secondaryBatchRaw.get(), "c_secondaryBatchRaw");
    dump(r_secondaryBatchRaw.get(), "r_secondaryBatchRaw");

    // the normalized correlation surfaces, e.g., 41 x 41 for a half search range of 20
    correlate(r_referenceBatchRaw.get(), r_secondaryBatchRaw.get(), r_corrBatchRaw.get(),
        cuCorrFreqDomain.get(), corrNormalizerRaw.get(), "r_corrBatchRaw");

    // the peaks, to the pixel
    cuArraysMaxloc2D(*r_corrBatchRaw, *offsetInit, *r_maxval, stream);
    // the covariance of the offsets
    cuEstimateVariance(*r_corrBatchRaw, *offsetInit, *r_maxval, r_referenceBatchRaw->size,
        1, *r_covValue, stream);
    // the SNR: the peak over the mean of the correlation surface around it
    cuArraysCopyExtractCorr(*r_corrBatchRaw, *r_corrBatchRawZoomIn, *i_corrBatchZoomInValid,
        *offsetInit, stream);
    cuArraysSumCorr(*r_corrBatchRawZoomIn, *i_corrBatchZoomInValid, *r_corrBatchSum,
        *i_corrBatchValidCount, stream);
    dump(r_maxval.get(), "r_maxval");
    dump(r_corrBatchRawZoomIn.get(), "r_corrBatchRawStatZoomIn");
    dump(i_corrBatchZoomInValid.get(), "i_corrBatchZoomInValid");
    dump(r_corrBatchSum.get(), "r_corrBatchSum");
    dump(i_corrBatchValidCount.get(), "i_corrBatchValidCount");
    cuEstimateSnr(*r_corrBatchSum, *i_corrBatchValidCount, *r_maxval, *r_snrValue, stream);
    dump(offsetInit.get(), "i_offsetInit");
    dump(r_snrValue.get(), "r_snrValue");
    dump(r_covValue.get(), "r_covValue");

    // ==== second pass, oversampled, around the first peak ====

    // the starting pixels of the secondary windows around the peaks, with a half search range of
    // halfZoomWindowSizeRaw (e.g., 4 pixels) instead of halfSearchRange*Raw
    cuDetermineSecondaryExtractOffset(*offsetInit, *maxLocShift,
        param.halfSearchRangeDownRaw, param.halfSearchRangeAcrossRaw,   // old range
        param.halfZoomWindowSizeRaw, param.halfZoomWindowSizeRaw,       // new range
        stream);
    dump(offsetInit.get(), "i_offsetInitAdjusted");
    dump(maxLocShift.get(), "i_maxLocShift");

    // the reference windows: deramped, oversampled, their amplitudes with the mean subtracted
    cuDeramp(param.derampMethod, *c_referenceBatchRaw, param.derampAxis, stream);
    dump(c_referenceBatchRaw.get(), "c_referenceBatchRawDeramped");
    referenceBatchOverSampler.execute(*c_referenceBatchRaw, *c_referenceBatchOverSampled);
    cuArraysAbs(*c_referenceBatchOverSampled, *r_referenceBatchOverSampled, stream);
    dump(c_referenceBatchOverSampled.get(), "c_referenceBatchOverSampled");
    dump(r_referenceBatchOverSampled.get(), "r_referenceBatchOverSampled");
    cuArraysSubtractMean(*r_referenceBatchOverSampled, stream);
    dump(r_referenceBatchOverSampled.get(), "r_referenceBatchOverSampledSubMean");

    // the secondary windows around the peaks: deramped, oversampled, their amplitudes
    cuArraysCopyExtract(*c_secondaryBatchRaw, *c_secondaryBatchZoomIn, *offsetInit, stream);
    dump(c_secondaryBatchZoomIn.get(), "c_secondaryBatchZoomInRaw");
    cuDeramp(param.derampMethod, *c_secondaryBatchZoomIn, param.derampAxis, stream);
    dump(c_secondaryBatchZoomIn.get(), "c_secondaryBatchZoomInDeramped");
    secondaryBatchOverSampler.execute(*c_secondaryBatchZoomIn, *c_secondaryBatchOverSampled);
    cuArraysAbs(*c_secondaryBatchOverSampled, *r_secondaryBatchOverSampled, stream);
    dump(c_secondaryBatchOverSampled.get(), "c_secondaryBatchOverSampled");
    dump(r_secondaryBatchOverSampled.get(), "r_secondaryBatchOverSampled");

    // the normalized correlation surfaces of the oversampled windows
    correlate(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatchZoomIn.get(),
        cuCorrFreqDomain_OverSampled.get(), corrNormalizerOverSampled.get(), "r_corrBatchZoomIn");

    // without the last row and column, of even size, oversampled again around the peak
    cuArraysCopyExtract(*r_corrBatchZoomIn, *r_corrBatchZoomInAdjust, traits::make_int2(0, 0), stream);
    dump(r_corrBatchZoomInAdjust.get(), "r_corrBatchZoomInAdjust");
    oversampleSurface(r_corrBatchZoomInAdjust.get(), r_corrBatchZoomInOverSampled.get(), maxLocShift.get());

    // the peaks of the oversampled surfaces
    cuArraysMaxloc2D(*r_corrBatchZoomInOverSampled, *offsetZoomIn, *corrMaxValue, stream);
    dump(offsetZoomIn.get(), "i_offsetZoomIn");
    dump(corrMaxValue.get(), "r_maxvalZoomInOversampled");

    // the offsets: the pixel peak (relative to the half search range) plus the sub-pixel peak,
    // (Init - HalfSearchRange) + ZoomIn/(2*ovs)
    cuSubPixelOffset2Pass(*offsetInit, *offsetZoomIn, *offsetFinal,
        param.corrSurfaceOverSamplingFactor, param.rawDataOversamplingFactor,
        param.halfSearchRangeDownRaw, param.halfSearchRangeAcrossRaw, stream);

    writeResults();
}

/**
 * A processor writing to the output images: allocate its buffers and workers
 */
template <template <typename> class arrayT>
cuAmpcorProcessorTwoPass<arrayT>::cuAmpcorProcessorTwoPass(const cuAmpcorParameter &param_,
    array_type<real2_type> &offsetImage, array_type<real_type> &snrImage,
    array_type<real3_type> &covImage, array_type<real_type> &peakValueImage, stream_type stream_)
    : base_type(param_, offsetImage, snrImage, covImage, peakValueImage, stream_),
      // the reference windows, from the raw size to the oversampled size
      referenceBatchOverSampler(param_.windowSizeHeightRaw, param_.windowSizeWidthRaw,
          param_.windowSizeHeight, param_.windowSizeWidth,
          param_.windowsDownInChunk*param_.windowsAcrossInChunk, stream_),
      // the secondary windows around the first peak, from the raw size to the oversampled size
      secondaryBatchOverSampler(param_.searchWindowSizeHeightRawZoomIn, param_.searchWindowSizeWidthRawZoomIn,
          param_.searchWindowSizeHeight, param_.searchWindowSizeWidth,
          param_.windowsDownInChunk*param_.windowsAcrossInChunk, stream_)
{
    using traits = typename base_type::traits;
    const int down = param.windowsDownInChunk;
    const int across = param.windowsAcrossInChunk;
    // a batch of windows of height x width (one per window in a chunk)
    auto batch = [down, across](auto &array, int height, int width) {
        using array_t = typename std::remove_reference_t<decltype(array)>::element_type;
        array = std::make_unique<array_t>(height, width, down, across);
        array->allocate();
    };

    // first pass
    batch(c_referenceBatchRaw, param.windowSizeHeightRaw, param.windowSizeWidthRaw);
    batch(c_secondaryBatchRaw, param.searchWindowSizeHeightRaw, param.searchWindowSizeWidthRaw);
    batch(r_referenceBatchRaw, param.windowSizeHeightRaw, param.windowSizeWidthRaw);
    batch(r_secondaryBatchRaw, param.searchWindowSizeHeightRaw, param.searchWindowSizeWidthRaw);
    batch(r_corrBatchRaw, param.searchWindowSizeHeightRaw - param.windowSizeHeightRaw + 1,
        param.searchWindowSizeWidthRaw - param.windowSizeWidthRaw + 1);

    // second pass
    batch(c_secondaryBatchZoomIn, param.searchWindowSizeHeightRawZoomIn, param.searchWindowSizeWidthRawZoomIn);
    maxLocShift = std::make_unique<array_type<int2_type>>(down, across);
    maxLocShift->allocate();
    batch(c_referenceBatchOverSampled, param.windowSizeHeight, param.windowSizeWidth);
    batch(c_secondaryBatchOverSampled, param.searchWindowSizeHeight, param.searchWindowSizeWidth);
    batch(r_referenceBatchOverSampled, param.windowSizeHeight, param.windowSizeWidth);
    batch(r_secondaryBatchOverSampled, param.searchWindowSizeHeight, param.searchWindowSizeWidth);
    batch(r_corrBatchZoomIn, param.searchWindowSizeHeight - param.windowSizeHeight + 1,
        param.searchWindowSizeWidth - param.windowSizeWidth + 1);
    batch(r_corrBatchZoomInAdjust, param.searchWindowSizeHeight - param.windowSizeHeight,
        param.searchWindowSizeWidth - param.windowSizeWidth);
    batch(r_corrBatchZoomInOverSampled, param.corrZoomWindowSize * param.corrSurfaceOverSamplingFactor,
        param.corrZoomWindowSize * param.corrSurfaceOverSamplingFactor);
    createSurfaceOversampler(param.corrZoomWindowSize, param.corrZoomWindowSize);

    // the correlators (frequency domain) and the normalizers, of both passes
    if (param.algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain = std::make_unique<freq_correlator_type>(
            param.searchWindowSizeHeightRaw, param.searchWindowSizeWidthRaw, down*across, stream);
        cuCorrFreqDomain_OverSampled = std::make_unique<freq_correlator_type>(
            param.searchWindowSizeHeight, param.searchWindowSizeWidth, down*across, stream);
    }
    corrNormalizerRaw = traits::newNormalizer(param.searchWindowSizeHeightRaw,
        param.searchWindowSizeWidthRaw, down*across);
    corrNormalizerOverSampled = traits::newNormalizer(param.searchWindowSizeHeight,
        param.searchWindowSizeWidth, down*across);

    logDebug("all objects in chunk are created");
}

/// the buffers and workers are released with their owners
template <template <typename> class arrayT>
cuAmpcorProcessorTwoPass<arrayT>::~cuAmpcorProcessorTwoPass() = default;

} // namespace
//end of file
