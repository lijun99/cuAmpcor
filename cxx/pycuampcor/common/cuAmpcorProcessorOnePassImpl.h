/**
 * @file  cuAmpcorProcessorOnePassImpl.h
 * @brief The one-pass workflow: the implementation
 *
 * The definitions of the member functions of cuAmpcorProcessorOnePass (templates); only the packages'
 * processors.cpp include them (see cuAmpcorProcessorImpl.h).
 */

#pragma once

#include "cuAmpcorProcessorOnePass.h"
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
void cuAmpcorProcessorOnePass<arrayT>::run(int idxDown, int idxAcross, const chunk_type &chunk)
{
    using traits = typename base_type::traits;
    setIndex(idxDown, idxAcross);

    // the reference windows (enlarged to the secondary size): deramped, oversampled, and the
    // amplitudes of their center (of the reference size), with the mean subtracted
    copyToBatch(chunk.reference, param.referenceStartPixelDown, param.referenceStartPixelAcross,
        c_referenceBatchRaw.get());
    dump(c_referenceBatchRaw.get(), "c_referenceBatchRaw");
    cuDeramp(param.derampMethod, c_referenceBatchRaw.get(), param.derampAxis, stream);
    dump(c_referenceBatchRaw.get(), "c_referenceBatchRawDeramped");
    referenceBatchOverSampler.execute(c_referenceBatchRaw.get(), c_referenceBatchOverSampled.get());
    const auto center = traits::make_int2(
        (c_referenceBatchOverSampled->height - r_referenceBatchOverSampled->height)/2,
        (c_referenceBatchOverSampled->width - r_referenceBatchOverSampled->width)/2);
    cuArraysCopyExtractAbs(c_referenceBatchOverSampled.get(), r_referenceBatchOverSampled.get(), center, stream);
    dump(c_referenceBatchOverSampled.get(), "c_referenceBatchOverSampled");
    dump(r_referenceBatchOverSampled.get(), "r_referenceBatchOverSampled");
    cuArraysSubtractMean(r_referenceBatchOverSampled.get(), stream);
    dump(r_referenceBatchOverSampled.get(), "r_referenceBatchOverSampledSubMean");

    // the secondary windows: deramped, oversampled, their amplitudes
    copyToBatch(chunk.secondary, param.secondaryStartPixelDown, param.secondaryStartPixelAcross,
        c_secondaryBatchRaw.get());
    dump(c_secondaryBatchRaw.get(), "c_secondaryBatchRaw");
    cuDeramp(param.derampMethod, c_secondaryBatchRaw.get(), param.derampAxis, stream);
    dump(c_secondaryBatchRaw.get(), "c_secondaryBatchRawDeramped");
    secondaryBatchOverSampler.execute(c_secondaryBatchRaw.get(), c_secondaryBatchOverSampled.get());
    cuArraysAbs(c_secondaryBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), stream);
    dump(c_secondaryBatchOverSampled.get(), "c_secondaryBatchOverSampled");
    dump(r_secondaryBatchOverSampled.get(), "r_secondaryBatchOverSampled");

    // the normalized correlation surfaces of the oversampled windows
    correlate(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatch.get(),
        cuCorrFreqDomain_OverSampled.get(), corrNormalizerOverSampled.get(), "r_corrBatch");

    // the peaks, within the search range: without the extra margin (for the surfaces around the peak)
    const int extraPadSize = param.halfZoomWindowSizeRaw*param.rawDataOversamplingFactor;
    const auto start = traits::make_int2(extraPadSize, extraPadSize);
    const auto range = traits::make_int2(r_corrBatch->height - 2*extraPadSize, r_corrBatch->width - 2*extraPadSize);
    cuArraysMaxloc2D(r_corrBatch.get(), start, range, offsetInit.get(), r_maxval.get(), stream);
    dump(offsetInit.get(), "i_offsetInit");
    dump(r_maxval.get(), "r_maxvalInit");

    // the surfaces around the peaks, for the oversampling (within the margin: all valid)
    cuArraysCopyExtractCorr(r_corrBatch.get(), r_corrBatchZoomIn.get(), offsetInit.get(), stream);
    dump(r_corrBatchZoomIn.get(), "r_corrBatchZoomIn");

    // the covariance of the offsets
    cuEstimateVariance(r_corrBatch.get(), offsetInit.get(), r_maxval.get(), r_referenceBatchOverSampled->size,
        param.rawDataOversamplingFactor, r_covValue.get(), stream);
    // the SNR, as in the two-pass workflow: the peak over the mean of the correlation surface
    // around it (corrStatWindowSize), at the raw pixel spacing and within the search range
    cuArraysCopyExtractCorr(r_corrBatch.get(), r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(),
        offsetInit.get(), param.rawDataOversamplingFactor, start, range, stream);
    cuArraysSumCorr(r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(), r_corrBatchSum.get(),
        i_corrBatchValidCount.get(), stream);
    cuEstimateSnr(r_corrBatchSum.get(), i_corrBatchValidCount.get(), r_maxval.get(), r_snrValue.get(), stream);
    dump(r_snrValue.get(), "r_snrValue");
    dump(r_covValue.get(), "r_covValue");

    // the oversampled surfaces around the peaks, and their peaks within +/- 1 pixel of the center
    // (the surfaces are extracted around their peaks: no shift)
    oversampleSurface(r_corrBatchZoomIn.get(), r_corrBatchZoomInOverSampled.get(), nullptr);
    const auto searchStart = traits::make_int2(param.corrZoomInOversampledSearchStart.x,
        param.corrZoomInOversampledSearchStart.y);
    const auto searchRange = traits::make_int2(param.corrZoomInOversampledSearchRange.x,
        param.corrZoomInOversampledSearchRange.y);
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled.get(), searchStart, searchRange,
        offsetZoomIn.get(), corrMaxValue.get(), stream);
    dump(offsetZoomIn.get(), "i_offsetZoomIn");
    dump(corrMaxValue.get(), "r_maxvalZoomInOversampled");

    // the offsets: the peak of the surfaces (at the oversampled spacing) plus the sub-pixel peak
    cuSubPixelOffset(offsetInit.get(), offsetZoomIn.get(), offsetFinal.get(),
        traits::make_int2(param.corrWindowSize.x/2, param.corrWindowSize.y/2),   // the origin of the first peak
        param.rawDataOversamplingFactor,                                           // its oversampling
        traits::make_int2(param.corrZoomInSize.x/2*param.corrSurfaceOverSamplingFactor,
                          param.corrZoomInSize.y/2*param.corrSurfaceOverSamplingFactor),  // the origin of the sub-pixel peak
        param.rawDataOversamplingFactor*param.corrSurfaceOverSamplingFactor,      // its oversampling
        stream);
    dump(offsetFinal.get(), "i_offsetFinal");

    writeResults();
}

/**
 * A processor writing to the output images: allocate its buffers and workers
 */
template <template <typename> class arrayT>
cuAmpcorProcessorOnePass<arrayT>::cuAmpcorProcessorOnePass(const cuAmpcorParameter &param_,
    array_type<real2_type> &offsetImage, array_type<real_type> &snrImage,
    array_type<real3_type> &covImage, array_type<real_type> &peakValueImage, stream_type stream_)
    : base_type(param_, offsetImage, snrImage, covImage, peakValueImage, stream_),
      // the reference windows (enlarged), from the raw size to the oversampled size
      referenceBatchOverSampler(param_.windowSizeHeightRawEnlarged, param_.windowSizeWidthRawEnlarged,
          param_.windowSizeHeightEnlarged, param_.windowSizeWidthEnlarged,
          param_.windowsDownInChunk*param_.windowsAcrossInChunk, stream_),
      // the secondary windows, from the raw size to the oversampled size
      secondaryBatchOverSampler(param_.searchWindowSizeHeightRaw, param_.searchWindowSizeWidthRaw,
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

    // the windows, and their oversampled amplitudes
    batch(c_referenceBatchRaw, param.windowSizeHeightRawEnlarged, param.windowSizeWidthRawEnlarged);
    batch(c_secondaryBatchRaw, param.searchWindowSizeHeightRaw, param.searchWindowSizeWidthRaw);
    batch(c_referenceBatchOverSampled, param.windowSizeHeightEnlarged, param.windowSizeWidthEnlarged);
    batch(c_secondaryBatchOverSampled, param.searchWindowSizeHeight, param.searchWindowSizeWidth);
    batch(r_referenceBatchOverSampled, param.windowSizeHeight, param.windowSizeWidth);
    batch(r_secondaryBatchOverSampled, param.searchWindowSizeHeight, param.searchWindowSizeWidth);

    // the correlation surfaces
    batch(r_corrBatch, param.corrWindowSize.x, param.corrWindowSize.y);
    batch(r_corrBatchZoomIn, param.corrZoomInSize.x, param.corrZoomInSize.y);
    batch(r_corrBatchZoomInOverSampled, param.corrZoomInOversampledSize.x, param.corrZoomInOversampledSize.y);
    createSurfaceOversampler(r_corrBatchZoomIn->height, r_corrBatchZoomIn->width);

    // the correlator (frequency domain) and the normalizer
    if (param.algorithm == Algorithm::Frequency)
        cuCorrFreqDomain_OverSampled = std::make_unique<freq_correlator_type>(
            param.searchWindowSizeHeight, param.searchWindowSizeWidth, down*across, stream);
    corrNormalizerOverSampled = traits::newNormalizer(param.searchWindowSizeHeight,
        param.searchWindowSizeWidth, down*across);

    logDebug("all objects in chunk are created");
}

/// the buffers and workers are released with their owners
template <template <typename> class arrayT>
cuAmpcorProcessorOnePass<arrayT>::~cuAmpcorProcessorOnePass() = default;

} // namespace
//end of file
