/**
 * @file  cuAmpcorProcessorImpl.h
 * @brief The processor of a chunk: the implementation of the base of the workflows
 *
 * The definitions of the member functions of cuAmpcorProcessor (templates). Only the packages'
 * processors.cpp include them, after their arrays, kernels and array_traits, and instantiate the
 * processors for their arrays: the processors are compiled once per package, and the other code
 * (e.g., the controllers) needs only the declarations (cuAmpcorProcessor.h).
 */

#pragma once

#include "cuAmpcorProcessor.h"
#include "cuAmpcorProcessorTwoPass.h"
#include "cuAmpcorProcessorOnePass.h"
#include "cuAmpcorOptions.h"
#include "cuAmpcorParameter.h"
#include "cuNormalizer.h"
#include "cuSurfaceOverSampler.h"

namespace pycuampcor {

/**
 * The processor of a workflow
 * @param workflow the workflow: two-pass or one-pass
 * @return the processor
 */
template <template <typename> class arrayT>
std::unique_ptr<cuAmpcorProcessor<arrayT>> cuAmpcorProcessor<arrayT>::create(Workflow workflow,
    const cuAmpcorParameter &param,
    array_type<real2_type> &offsetImage, array_type<real_type> &snrImage,
    array_type<real3_type> &covImage, array_type<real_type> &peakValueImage,
    stream_type stream)
{
    switch (workflow) {
        case Workflow::TwoPass:
            return std::make_unique<cuAmpcorProcessorTwoPass<arrayT>>(
                param, offsetImage, snrImage, covImage, peakValueImage, stream);
        case Workflow::OnePass:
            return std::make_unique<cuAmpcorProcessorOnePass<arrayT>>(
                param, offsetImage, snrImage, covImage, peakValueImage, stream);
        default:
            throw std::invalid_argument("Unsupported workflow");
    }
}

/**
 * A processor writing to the output images: allocate the buffers of the per-window results
 */
template <template <typename> class arrayT>
cuAmpcorProcessor<arrayT>::cuAmpcorProcessor(const cuAmpcorParameter &param_,
    array_type<real2_type> &offsetImage_, array_type<real_type> &snrImage_,
    array_type<real3_type> &covImage_, array_type<real_type> &peakValueImage_,
    stream_type stream_)
    : param(param_), offsetImage(offsetImage_), snrImage(snrImage_), covImage(covImage_),
      peakValueImage(peakValueImage_), stream(stream_)
{
    const int down = param.windowsDownInChunk;
    const int across = param.windowsAcrossInChunk;
    // a batch of count windows of height x width (one per window in a chunk), in the work memory
    auto batch = [down, across](auto &array, int height, int width) {
        using array_t = typename std::remove_reference_t<decltype(array)>::element_type;
        array = std::make_unique<array_t>(height, width, down, across);
        array->allocate();
    };
    // one value per window in a chunk
    auto values = [down, across](auto &array) {
        using array_t = typename std::remove_reference_t<decltype(array)>::element_type;
        array = std::make_unique<array_t>(down, across);
        array->allocate();
    };

    // the starting pixels of the windows relative to the loaded chunk, computed on the host
    values(chunkOffsetDown);
    chunkOffsetDown->allocateHost();
    values(chunkOffsetAcross);
    chunkOffsetAcross->allocateHost();

    // the per-window results
    values(offsetInit);
    values(offsetZoomIn);
    values(offsetFinal);
    values(r_maxval);
    values(corrMaxValue);
    values(r_covValue);
    values(r_snrValue);
    values(r_corrBatchSum);
    values(i_corrBatchValidCount);
    // the correlation surfaces around the peaks, at the raw pixel spacing, for the SNR
    batch(r_corrBatchRawZoomIn, param.corrRawZoomInHeight, param.corrRawZoomInWidth);
    batch(i_corrBatchZoomInValid, param.corrRawZoomInHeight, param.corrRawZoomInWidth);
}

/// the buffers and workers are released with their owners
template <template <typename> class arrayT>
cuAmpcorProcessor<arrayT>::~cuAmpcorProcessor() = default;

/**
 * Set the index of the current chunk, and its number of windows (fewer in the last chunks)
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::setIndex(int idxDown, int idxAcross)
{
    idxChunkDown = idxDown;
    idxChunkAcross = idxAcross;
    nWindowsDown = (idxChunkDown == param.numberChunkDown - 1)
        ? param.numberWindowDown - param.windowsDownInChunk*(param.numberChunkDown - 1)
        : param.windowsDownInChunk;
    nWindowsAcross = (idxChunkAcross == param.numberChunkAcross - 1)
        ? param.numberWindowAcross - param.windowsAcrossInChunk*(param.numberChunkAcross - 1)
        : param.windowsAcrossInChunk;
}

/**
 * The starting pixels of the windows of the current chunk, relative to the loaded chunk
 * (the windows beyond the last ones of the chunk repeat the last ones)
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::getRelativeOffset(int *relative, const std::vector<int> &absolute,
    int chunkStart)
{
    for (int i = 0; i < param.windowsDownInChunk; ++i) {
        const int iDown = std::min(i, nWindowsDown - 1);
        for (int j = 0; j < param.windowsAcrossInChunk; ++j) {
            const int iAcross = std::min(j, nWindowsAcross - 1);
            const int idxInChunk = iDown*param.windowsAcrossInChunk + iAcross;
            const int idxInAll = (iDown + idxChunkDown*param.windowsDownInChunk)*param.numberWindowAcross
                + idxChunkAcross*param.windowsAcrossInChunk + iAcross;
            relative[idxInChunk] = absolute[idxInAll] - chunkStart;
        }
    }
}

/**
 * Copy the windows of the current chunk from a loaded chunk to a batch
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::copyToBatch(const loaded_chunk_type &chunk,
    const std::vector<int> &startDown, const std::vector<int> &startAcross,
    array_type<complex_type> *batch)
{
    // a chunk entirely outside the image: its windows are zeros
    if (chunk.empty()) {
        batch->setZero(stream);
        return;
    }

    // the starting pixels of the windows relative to the chunk, computed on the host
    getRelativeOffset(chunkOffsetDown->hostData, startDown, chunk.startDown);
    chunkOffsetDown->copyToDevice(stream);
    getRelativeOffset(chunkOffsetAcross->hostData, startAcross, chunk.startAcross);
    chunkOffsetAcross->copyToDevice(stream);

    // windows outside the chunk (image) are padded with zeros
    if (!chunk.complexData.empty()) {
        // complex image (e.g., SLC): amplitudes if not deramped, the complex values otherwise
        if (param.derampMethod == DerampMethod::Magnitude)
            cuArraysCopyToBatchAbsWithOffset(chunk.complexData, *batch, *chunkOffsetDown,
                *chunkOffsetAcross, stream);
        else
            cuArraysCopyToBatchWithOffset(chunk.complexData, *batch, *chunkOffsetDown,
                *chunkOffsetAcross, stream);
    }
    else {
        // real image (e.g., TIFF), copied to complex
        cuArraysCopyToBatchWithOffsetR2C(chunk.realData, *batch, *chunkOffsetDown,
            *chunkOffsetAcross, stream);
    }
}

/**
 * Cross-correlate the reference windows with the secondary windows, and normalize
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::correlate(array_type<real_type> *reference,
    array_type<real_type> *secondary, array_type<real_type> *surface,
    freq_correlator_type *freqCorrelator, normalizer_type *normalizer, const std::string &name)
{
    if (param.algorithm == Algorithm::Frequency)
        freqCorrelator->execute(reference, secondary, surface);
    else
        cuCorrTimeDomain(reference, secondary, surface, stream);
    dump(surface, name + "UnNorm");

    normalizer->execute(surface, reference, secondary, stream);
    dump(surface, name);
}

/**
 * Create the oversampler of the correlation surfaces of height x width around the peak
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::createSurfaceOversampler(int height, int width)
{
    surfaceOverSampler = traits::newSurfaceOverSampler(param.corrSurfaceOverSamplingMethod,
        height, width, param.corrSurfaceOverSamplingFactor, param.rawDataOversamplingFactor,
        param.windowsDownInChunk*param.windowsAcrossInChunk, stream);
}

/**
 * Oversample the correlation surfaces around the peak
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::oversampleSurface(array_type<real_type> *surface,
    array_type<real_type> *oversampled, array_type<int2_type> *shift)
{
    surfaceOverSampler->execute(surface, oversampled, shift);
    dump(oversampled, "r_corrBatchZoomInOverSampled");
}

/**
 * Write the results of the current chunk to the output images
 */
template <template <typename> class arrayT>
void cuAmpcorProcessor<arrayT>::writeResults()
{
    // the first window of the chunk in the output images
    const int down = idxChunkDown*param.windowsDownInChunk;
    const int across = idxChunkAcross*param.windowsAcrossInChunk;
    cuArraysCopyInsert(*offsetFinal, offsetImage, down, across, stream);
    cuArraysCopyInsert(*r_snrValue, snrImage, down, across, stream);
    cuArraysCopyInsert(*r_covValue, covImage, down, across, stream);
    cuArraysCopyInsert(*corrMaxValue, peakValueImage, down, across, stream);
}

/**
 * Write an intermediate array to a file (debug builds only)
 */
template <template <typename> class arrayT>
template <typename T>
void cuAmpcorProcessor<arrayT>::dump([[maybe_unused]] array_type<T> *array,
    [[maybe_unused]] const std::string &name)
{
#ifdef CUAMPCOR_DEBUG
    array->outputToFile(name, stream);
#endif
}

} // namespace
//end of file
