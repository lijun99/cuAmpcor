#include "cuAmpcorProcessorOnePass.h"
#include "logging.h"
#include "cuArraysCopy.h"
#include "cuDeramp.h"
#include "cuCorrNormalization.h"
#include "cuOffset.h"
#include "cuCorrTimeDomain.h"
#include "cuEstimateStats.h"
#include "debug.h"

#include <iostream>

namespace pycuampcor::PYCUAMPCOR_BACKEND {

/**
 * Run ampcor process for a batch of images (a chunk)
 * @param[in] idxDown_  index of the chunk along Down/Azimuth direction
 * @param[in] idxAcross_ index of the chunk along Across/Range direction
 */
void cuAmpcorProcessorOnePass::run(int idxDown_, int idxAcross_, const cuAmpcorChunk &chunk)
{
    // set chunk index
    setIndex(idxDown_, idxAcross_);

    // copy the reference windows from the loaded chunk
    copyToBatch(chunk.reference, param->referenceStartPixelDown, param->referenceStartPixelAcross, c_referenceBatchRaw.get());

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRaw.get()", stream);
#endif

    // deramp reference
    cuDeramp(param->derampMethod, c_referenceBatchRaw.get(), param->derampAxis, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the deramped reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRawDeramped", stream);
#endif

    // oversample reference
    referenceBatchOverSampler->execute(c_referenceBatchRaw.get(), c_referenceBatchOverSampled.get());

    // offset to extract
    int2 offset = make_int2((c_referenceBatchOverSampled->height - r_referenceBatchOverSampled->height)/2,
        (c_referenceBatchOverSampled->width - r_referenceBatchOverSampled->width)/2);

    // extract and take amplitudes
    cuArraysCopyExtractAbs(c_referenceBatchOverSampled.get(), r_referenceBatchOverSampled.get(), offset, stream);

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

    // copy the secondary windows from the loaded chunk
    copyToBatch(chunk.secondary, param->secondaryStartPixelDown, param->secondaryStartPixelAcross, c_secondaryBatchRaw.get());

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_secondaryBatchRaw->outputToFile("c_secondaryBatchRaw.get()", stream);
#endif

    // deramp secondary image(s)
    cuDeramp(param->derampMethod, c_secondaryBatchRaw.get(), param->derampAxis, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_secondaryBatchRaw->outputToFile("c_secondaryBatchRawDeramped", stream);
#endif

    // oversampling the secondary image(s)
    secondaryBatchOverSampler->execute(c_secondaryBatchRaw.get(), c_secondaryBatchOverSampled.get());
    // take amplitudes
    cuArraysAbs(c_secondaryBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled secondary image(s)
    c_secondaryBatchOverSampled->outputToFile("c_secondaryBatchOverSampled.get()", stream);
    r_secondaryBatchOverSampled->outputToFile("r_secondaryBatchOverSampled.get()", stream);
#endif

    // correlate oversampled images
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain_OverSampled->execute(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatch.get());
    }
    else {
        cuCorrTimeDomain(r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), r_corrBatch.get(), stream);
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (un-normalized)
    r_corrBatch->outputToFile("r_corrBatch.get()", stream);
#endif

    // normalize the correlation surface
    corrNormalizerOverSampled->execute(r_corrBatch.get(), r_referenceBatchOverSampled.get(), r_secondaryBatchOverSampled.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (normalized)
    r_corrBatch->outputToFile("r_corrBatchNormed", stream);
#endif

    // find the maximum location of the correlation surface, in a rectangle area {range} from {start}
    int extraPadSize = param->halfZoomWindowSizeRaw*param->rawDataOversamplingFactor;
    int2 start = make_int2(extraPadSize, extraPadSize);
    int2 range = make_int2(r_corrBatch->height-2*extraPadSize, r_corrBatch->width-2*extraPadSize);
    cuArraysMaxloc2D(r_corrBatch.get(), start, range, offsetInit.get(), r_maxval.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the max location and value
    offsetInit->outputToFile("i_offsetInit", stream);
    r_maxval->outputToFile("r_maxvalInit", stream);
#endif

    // extract a smaller chip around the peak {offsetInit} for oversampling
    // (within the extra pads, so all its pixels are valid)
    cuArraysCopyExtractCorr(r_corrBatch.get(), r_corrBatchZoomIn.get(), offsetInit.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the extracted correlation Surface
    r_corrBatchZoomIn->outputToFile("r_corrBatchZoomIn.get()", stream);
#endif

    // statistics of correlation surface
    // estimate variance on r_corrBatch
    cuEstimateVariance(r_corrBatch.get(), offsetInit.get(), r_maxval.get(), r_referenceBatchOverSampled->size, param->rawDataOversamplingFactor, r_covValue.get(), stream);

    // snr as in the two-pass workflow: the peak over the mean of the correlation surface around it
    // (corrStatWindowSize), at the raw pixel spacing and within the search range
    cuArraysCopyExtractCorr(r_corrBatch.get(), r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(), offsetInit.get(),
        param->rawDataOversamplingFactor, start, range, stream);
    cuArraysSumCorr(r_corrBatchRawZoomIn.get(), i_corrBatchZoomInValid.get(), r_corrBatchSum.get(), i_corrBatchValidCount.get(), stream);
    cuEstimateSnr(r_corrBatchSum.get(), i_corrBatchValidCount.get(), r_maxval.get(), r_snrValue.get(), stream);

#ifdef CUAMPCOR_DEBUG
    r_snrValue->outputToFile("r_snrValue.get()", stream);
    r_covValue->outputToFile("r_covValue.get()", stream);
#endif

    // oversample the correlation surface
    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        // sinc interpolator only computes (-i_sincwindow, i_sincwindow)*oversamplingfactor
        // we need the max loc as the center if shifted
        corrSincOverSampler->execute(r_corrBatchZoomIn.get(), r_corrBatchZoomInOverSampled.get(),
             maxLocShift, param->corrSurfaceOverSamplingFactor*param->rawDataOversamplingFactor
            );

    }
    else {
        corrOverSampler->execute(r_corrBatchZoomIn.get(), r_corrBatchZoomInOverSampled.get());
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface
    r_corrBatchZoomInOverSampled->outputToFile("r_corrBatchZoomInOverSampled.get()", stream);
#endif

    //find the max again, within the range of \pm 1 pixel * totalOS
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled.get(), param->corrZoomInOversampledSearchStart, param->corrZoomInOversampledSearchRange,
        offsetZoomIn.get(), corrMaxValue.get(), stream);

#ifdef CUAMPCOR_DEBUG
    // dump the max location on oversampled correlation surface
    offsetZoomIn->outputToFile("i_offsetZoomIn", stream);
    corrMaxValue->outputToFile("r_maxvalZoomInOversampled", stream);
#endif

    // determine the final offset from initial (pixel/2) and oversampled (sub-pixel)
    cuSubPixelOffset(offsetInit.get(), offsetZoomIn.get(), offsetFinal.get(),
        make_int2(param->corrWindowSize.x/2, param->corrWindowSize.y/2), // init offset origin
        param->rawDataOversamplingFactor, // init offset factor
        make_int2(param->corrZoomInSize.x/2*param->corrSurfaceOverSamplingFactor, param->corrZoomInSize.y/2*param->corrSurfaceOverSamplingFactor),
        param->rawDataOversamplingFactor*param->corrSurfaceOverSamplingFactor,
        stream);

#ifdef CUAMPCOR_DEBUG
    // dump the final offset
    offsetFinal->outputToFile("i_offsetFinal", stream);
#endif

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
cuAmpcorProcessorOnePass::cuAmpcorProcessorOnePass(cuAmpcorParameter *param_,
    cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_, cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
    stream_t stream_)
    : cuAmpcorProcessor(param_, offsetImage_, snrImage_, covImage_, peakValueImage_, stream_)
{


    c_referenceBatchRaw = std::make_unique<cuArrays<complex_type>>(
        param->windowSizeHeightRawEnlarged, param->windowSizeWidthRawEnlarged,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_referenceBatchRaw->allocate();

    c_secondaryBatchRaw = std::make_unique<cuArrays<complex_type>>(
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchRaw->allocate();

    c_referenceBatchOverSampled = std::make_unique<cuArrays<complex_type>>(
            param->windowSizeHeightEnlarged, param->windowSizeWidthEnlarged,
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

    secondaryBatchOverSampler = std::make_unique<cuOverSamplerC2C>(
        c_secondaryBatchRaw->height, c_secondaryBatchRaw->width,
        c_secondaryBatchOverSampled->height, c_secondaryBatchOverSampled->width,
        c_secondaryBatchRaw->count, stream);

    r_corrBatch = std::make_unique<cuArrays<real_type>>(
        param->corrWindowSize.x,
        param->corrWindowSize.y,
        param->windowsDownInChunk,
        param->windowsAcrossInChunk);
    r_corrBatch->allocate();

    r_corrBatchZoomIn = std::make_unique<cuArrays<real_type>>(
            param->corrZoomInSize.x,
            param->corrZoomInSize.y,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchZoomIn->allocate();

    r_corrBatchZoomInOverSampled = std::make_unique<cuArrays<real_type>>(
        param->corrZoomInOversampledSize.x,
        param->corrZoomInOversampledSize.y,
        param->windowsDownInChunk,
        param->windowsAcrossInChunk);
    r_corrBatchZoomInOverSampled->allocate();

    offsetInit = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetInit->allocate();

    offsetZoomIn = std::make_unique<cuArrays<int2>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetZoomIn->allocate();

    offsetFinal = std::make_unique<cuArrays<real2_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetFinal->allocate();

    // the zoom-in correlation surface is extracted around the peak,
    // so the sinc interpolation center needs no extra shift
    maxLocShift = make_int2(0, 0);

    corrMaxValue = std::make_unique<cuArrays<real_type>>(param->windowsDownInChunk, param->windowsAcrossInChunk);
    corrMaxValue->allocate();

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
        corrOverSampler = std::make_unique<cuOverSamplerR2R>(
            r_corrBatchZoomIn->height, r_corrBatchZoomIn->width,
            r_corrBatchZoomInOverSampled->height, r_corrBatchZoomInOverSampled->width,
            param->windowsDownInChunk*param->windowsAcrossInChunk,
            stream);
    }
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain_OverSampled = std::make_unique<cuFreqCorrelator>(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk * param->windowsAcrossInChunk,
            stream);
    }


    corrNormalizerOverSampled =
        newCuNormalizer(
        param->searchWindowSizeHeight,
        param->searchWindowSizeWidth,
        param->windowsDownInChunk * param->windowsAcrossInChunk
        );


    logDebug("all objects in chunk are created");
}

// destructor
cuAmpcorProcessorOnePass::~cuAmpcorProcessorOnePass() = default;

// end of file


} // namespace