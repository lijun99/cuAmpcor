/**
 * @file  ProcessorOnePassImpl.h
 * @brief The one-pass workflow: the implementation
 *
 * The definitions of the member functions of ProcessorOnePass (templates); only the packages'
 * processors.cpp include them (see ProcessorImpl.h).
 */

#pragma once

#include "ProcessorOnePass.h"
#include "Options.h"
#include "Parameter.h"
#include "Normalizer.h"
#include "logging.h"

namespace pycuampcor {

/**
 * Process the chunk (idxDown, idxAcross), and write its results to the output images
 * @param idxDown,idxAcross the index of the chunk, down and across
 * @param chunk the chunk, loaded by the chunk loader
 */
template <template <typename> class arrayT>
void ProcessorOnePass<arrayT>::run(int idxDown, int idxAcross, const chunk_type &chunk)
{
    using traits = typename base_type::traits;
    setIndex(idxDown, idxAcross);

    // ==== the windows, oversampled ====

    // the reference windows (enlarged to the secondary size), from the loaded chunk (zeros outside
    // the image): deramped, oversampled, and the amplitudes of their center (of the reference
    // size), with the mean subtracted
    copyToBatch(chunk.reference, param.referenceStartPixelDown, param.referenceStartPixelAcross,
        c_referenceBatchRaw);
    // dump: write the intermediate arrays to files, in debug builds only (CUAMPCOR_DEBUG, defined
    // without NDEBUG); a no-op otherwise
    dump(c_referenceBatchRaw, "c_referenceBatchRaw");

    // the phase ramp removed, so that the spectrum is centered for the oversampling
    cuDeramp(param.derampMethod, c_referenceBatchRaw, param.derampAxis, stream);
    dump(c_referenceBatchRaw, "c_referenceBatchRawDeramped");

    // oversampled by rawDataOversamplingFactor (FFT, zero padding)
    referenceBatchOverSampler.execute(c_referenceBatchRaw, c_referenceBatchOverSampled);

    // the amplitudes of the center, of the (oversampled) reference size: oversampled as part of
    // the enlarged windows, the center is away from the edge effects of the FFT oversampling
    const auto center = traits::make_int2(
        (c_referenceBatchOverSampled.height() - r_referenceBatchOverSampled.height())/2,
        (c_referenceBatchOverSampled.width() - r_referenceBatchOverSampled.width())/2);
    cuArraysCopyExtractAbs(c_referenceBatchOverSampled, r_referenceBatchOverSampled, center, stream);
    dump(c_referenceBatchOverSampled, "c_referenceBatchOverSampled");
    dump(r_referenceBatchOverSampled, "r_referenceBatchOverSampled");

    // the mean subtracted (for the normalized correlation)
    cuArraysSubtractMean(r_referenceBatchOverSampled, stream);
    dump(r_referenceBatchOverSampled, "r_referenceBatchOverSampledSubMean");

    // the secondary windows, from the loaded chunk (zeros outside the image): deramped,
    // oversampled, their amplitudes
    copyToBatch(chunk.secondary, param.secondaryStartPixelDown, param.secondaryStartPixelAcross,
        c_secondaryBatchRaw);
    dump(c_secondaryBatchRaw, "c_secondaryBatchRaw");

    // the phase ramp removed
    cuDeramp(param.derampMethod, c_secondaryBatchRaw, param.derampAxis, stream);
    dump(c_secondaryBatchRaw, "c_secondaryBatchRawDeramped");

    // oversampled by rawDataOversamplingFactor
    secondaryBatchOverSampler.execute(c_secondaryBatchRaw, c_secondaryBatchOverSampled);

    // their amplitudes
    cuArraysAbs(c_secondaryBatchOverSampled, r_secondaryBatchOverSampled, stream);
    dump(c_secondaryBatchOverSampled, "c_secondaryBatchOverSampled");
    dump(r_secondaryBatchOverSampled, "r_secondaryBatchOverSampled");

    // ==== the correlation surfaces, their peaks and statistics ====

    // the normalized correlation surfaces of the oversampled windows, over the search range
    // plus the margin (halfZoomWindowSizeRaw raw pixels each way) for the surfaces around the peaks
    correlate(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatch,
        cuCorrFreqDomain_OverSampled.get(), corrNormalizerOverSampled.get(), "r_corrBatch");

    // the peaks, to the oversampled pixel, within the search range: without the extra margin (for
    // the surfaces around the peak)
    const int extraPadSize = param.halfZoomWindowSizeRaw*param.rawDataOversamplingFactor;
    const auto start = traits::make_int2(extraPadSize, extraPadSize);
    const auto range = traits::make_int2(r_corrBatch.height() - 2*extraPadSize, r_corrBatch.width() - 2*extraPadSize);
    cuArraysMaxloc2D(r_corrBatch, start, range, offsetInit, r_maxval, stream);
    dump(offsetInit, "i_offsetInit");
    dump(r_maxval, "r_maxvalInit");

    // the surfaces around the peaks (corrZoomInSize), for the oversampling (within the margin:
    // all valid)
    cuArraysCopyExtractCorr(r_corrBatch, r_corrBatchZoomIn, offsetInit, stream);
    dump(r_corrBatchZoomIn, "r_corrBatchZoomIn");

    // the covariance of the offsets, from the curvature of the surfaces at the peaks (at the raw
    // pixel spacing: rawDataOversamplingFactor oversampled pixels apart)
    cuEstimateVariance(r_corrBatch, offsetInit, r_maxval, r_referenceBatchOverSampled.arraySize(),
        param.rawDataOversamplingFactor, r_covValue, stream);

    // the SNR, as in the two-pass workflow: the peak over the mean of the correlation surface
    // around it (corrStatWindowSize), at the raw pixel spacing and within the search range;
    // first, the surfaces around the peaks at the raw pixel spacing, their pixels within the
    // search range marked as valid
    cuArraysCopyExtractCorr(r_corrBatch, r_corrBatchRawZoomIn, i_corrBatchZoomInValid,
        offsetInit, param.rawDataOversamplingFactor, start, range, stream);

    // the sum of their squares, and the number of their valid pixels
    cuArraysSumCorr(r_corrBatchRawZoomIn, i_corrBatchZoomInValid, r_corrBatchSum,
        i_corrBatchValidCount, stream);

    // the squared peak over the mean of the squares of the other pixels
    cuEstimateSnr(r_corrBatchSum, i_corrBatchValidCount, r_maxval, r_snrValue, stream);
    dump(r_snrValue, "r_snrValue");
    dump(r_covValue, "r_covValue");

    // ==== the sub-pixel peaks ====

    // the surfaces around the peaks, oversampled by corrSurfaceOverSamplingFactor (the surfaces
    // are extracted around their peaks: no shift)
    oversampleSurface(r_corrBatchZoomIn, r_corrBatchZoomInOverSampled, {});

    // the peaks of the oversampled surfaces, within +/- 1 raw pixel of the center
    const auto searchStart = traits::make_int2(param.corrZoomInOversampledSearchStart.x,
        param.corrZoomInOversampledSearchStart.y);
    const auto searchRange = traits::make_int2(param.corrZoomInOversampledSearchRange.x,
        param.corrZoomInOversampledSearchRange.y);
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled, searchStart, searchRange,
        offsetZoomIn, corrMaxValue, stream);
    dump(offsetZoomIn, "i_offsetZoomIn");
    dump(corrMaxValue, "r_maxvalZoomInOversampled");

    // the offsets: the peak of the surfaces (at the oversampled spacing) plus the sub-pixel peak,
    // both relative to the centers of their surfaces, in raw pixels
    cuSubPixelOffset(offsetInit, offsetZoomIn, offsetFinal,
        traits::make_int2(param.corrWindowSize.x/2, param.corrWindowSize.y/2),   // the origin of the first peak
        param.rawDataOversamplingFactor,                                           // its oversampling
        traits::make_int2(param.corrZoomInSize.x/2*param.corrSurfaceOverSamplingFactor,
                          param.corrZoomInSize.y/2*param.corrSurfaceOverSamplingFactor),  // the origin of the sub-pixel peak
        param.rawDataOversamplingFactor*param.corrSurfaceOverSamplingFactor,      // its oversampling
        stream);
    dump(offsetFinal, "i_offsetFinal");

    // the offsets, SNR, covariance and peaks to the output images
    writeResults();
}

/**
 * A processor writing to the output images: allocate its buffers and workers
 */
template <template <typename> class arrayT>
ProcessorOnePass<arrayT>::ProcessorOnePass(const Parameter &param_,
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
        array = std::decay_t<decltype(array)>(height, width, down*across);
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
    createSurfaceOversampler(r_corrBatchZoomIn.height(), r_corrBatchZoomIn.width());

    // the correlator (frequency domain) and the normalizer
    if (param.algorithm == Algorithm::Frequency)
        cuCorrFreqDomain_OverSampled = std::make_unique<freq_correlator_type>(
            param.searchWindowSizeHeight, param.searchWindowSizeWidth, down*across, stream);
    corrNormalizerOverSampled = traits::newNormalizer(param.searchWindowSizeHeight,
        param.searchWindowSizeWidth, down*across);

    logDebug("all objects in chunk are created");
}

/**
 * Release the buffers and workers (they are released with their owners)
 */
template <template <typename> class arrayT>
ProcessorOnePass<arrayT>::~ProcessorOnePass() = default;

} // namespace
//end of file
