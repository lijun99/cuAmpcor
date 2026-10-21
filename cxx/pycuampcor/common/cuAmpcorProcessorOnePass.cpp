#include "cuAmpcorProcessorOnePass.h"

#include "cuAmpcorUtil.h"
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
    copyToBatch(chunk.reference, param->referenceStartPixelDown, param->referenceStartPixelAcross, c_referenceBatchRaw);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRaw", stream);
#endif

    // deramp reference
    cuDeramp(param->derampMethod, c_referenceBatchRaw, param->derampAxis, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the deramped reference image(s)
    c_referenceBatchRaw->outputToFile("c_referenceBatchRawDeramped", stream);
#endif

    // oversample reference
    referenceBatchOverSampler->execute(c_referenceBatchRaw, c_referenceBatchOverSampled);

    // offset to extract
    int2 offset = make_int2((c_referenceBatchOverSampled->height - r_referenceBatchOverSampled->height)/2,
        (c_referenceBatchOverSampled->width - r_referenceBatchOverSampled->width)/2);

    // extract and take amplitudes
    cuArraysCopyExtractAbs(c_referenceBatchOverSampled, r_referenceBatchOverSampled, offset, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled reference image(s)
    c_referenceBatchOverSampled->outputToFile("c_referenceBatchOverSampled", stream);
    r_referenceBatchOverSampled->outputToFile("r_referenceBatchOverSampled", stream);
#endif

    // compute and subtract the mean value
    cuArraysSubtractMean(r_referenceBatchOverSampled, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled reference image(s) with mean subtracted
    r_referenceBatchOverSampled->outputToFile("r_referenceBatchOverSampledSubMean",stream);
#endif

    // copy the secondary windows from the loaded chunk
    copyToBatch(chunk.secondary, param->secondaryStartPixelDown, param->secondaryStartPixelAcross, c_secondaryBatchRaw);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_secondaryBatchRaw->outputToFile("c_secondaryBatchRaw", stream);
#endif

    // deramp secondary image(s)
    cuDeramp(param->derampMethod, c_secondaryBatchRaw, param->derampAxis, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the raw reference image(s)
    c_secondaryBatchRaw->outputToFile("c_secondaryBatchRawDeramped", stream);
#endif

    // oversampling the secondary image(s)
    secondaryBatchOverSampler->execute(c_secondaryBatchRaw, c_secondaryBatchOverSampled);
    // take amplitudes
    cuArraysAbs(c_secondaryBatchOverSampled, r_secondaryBatchOverSampled, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled secondary image(s)
    c_secondaryBatchOverSampled->outputToFile("c_secondaryBatchOverSampled", stream);
    r_secondaryBatchOverSampled->outputToFile("r_secondaryBatchOverSampled", stream);
#endif

    // correlate oversampled images
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain_OverSampled->execute(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatch);
    }
    else {
        cuCorrTimeDomain(r_referenceBatchOverSampled, r_secondaryBatchOverSampled, r_corrBatch, stream);
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (un-normalized)
    r_corrBatch->outputToFile("r_corrBatch", stream);
#endif

    // normalize the correlation surface
    corrNormalizerOverSampled->execute(r_corrBatch, r_referenceBatchOverSampled, r_secondaryBatchOverSampled, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface (normalized)
    r_corrBatch->outputToFile("r_corrBatchNormed", stream);
#endif

    // find the maximum location of the correlation surface, in a rectangle area {range} from {start}
    int extraPadSize = param->halfZoomWindowSizeRaw*param->rawDataOversamplingFactor;
    int2 start = make_int2(extraPadSize, extraPadSize);
    int2 range = make_int2(r_corrBatch->height-2*extraPadSize, r_corrBatch->width-2*extraPadSize);
    cuArraysMaxloc2D(r_corrBatch, start, range, offsetInit, r_maxval, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the max location and value
    offsetInit->outputToFile("i_offsetInit", stream);
    r_maxval->outputToFile("r_maxvalInit", stream);
#endif

    // extract a smaller chip around the peak {offsetInit} for oversampling
    // (within the extra pads, so all its pixels are valid)
    cuArraysCopyExtractCorr(r_corrBatch, r_corrBatchZoomIn, offsetInit, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the extracted correlation Surface
    r_corrBatchZoomIn->outputToFile("r_corrBatchZoomIn", stream);
#endif

    // statistics of correlation surface
    // estimate variance on r_corrBatch
    cuEstimateVariance(r_corrBatch, offsetInit, r_maxval, r_referenceBatchOverSampled->size, param->rawDataOversamplingFactor, r_covValue, stream);

    // snr as in the two-pass workflow: the peak over the mean of the correlation surface around it
    // (corrStatWindowSize), at the raw pixel spacing and within the search range
    cuArraysCopyExtractCorr(r_corrBatch, r_corrBatchRawZoomIn, i_corrBatchZoomInValid, offsetInit,
        param->rawDataOversamplingFactor, start, range, stream);
    cuArraysSumCorr(r_corrBatchRawZoomIn, i_corrBatchZoomInValid, r_corrBatchSum, i_corrBatchValidCount, stream);
    cuEstimateSnr(r_corrBatchSum, i_corrBatchValidCount, r_maxval, r_snrValue, stream);

#ifdef CUAMPCOR_DEBUG
    r_snrValue->outputToFile("r_snrValue", stream);
    r_covValue->outputToFile("r_covValue", stream);
#endif

    // oversample the correlation surface
    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        // sinc interpolator only computes (-i_sincwindow, i_sincwindow)*oversamplingfactor
        // we need the max loc as the center if shifted
        corrSincOverSampler->execute(r_corrBatchZoomIn, r_corrBatchZoomInOverSampled,
             maxLocShift, param->corrSurfaceOverSamplingFactor*param->rawDataOversamplingFactor
            );

    }
    else {
        corrOverSampler->execute(r_corrBatchZoomIn, r_corrBatchZoomInOverSampled);
    }

#ifdef CUAMPCOR_DEBUG
    // dump the oversampled correlation surface
    r_corrBatchZoomInOverSampled->outputToFile("r_corrBatchZoomInOverSampled", stream);
#endif

    //find the max again, within the range of \pm 1 pixel * totalOS
    cuArraysMaxloc2D(r_corrBatchZoomInOverSampled, param->corrZoomInOversampledSearchStart, param->corrZoomInOversampledSearchRange,
        offsetZoomIn, corrMaxValue, stream);

#ifdef CUAMPCOR_DEBUG
    // dump the max location on oversampled correlation surface
    offsetZoomIn->outputToFile("i_offsetZoomIn", stream);
    corrMaxValue->outputToFile("r_maxvalZoomInOversampled", stream);
#endif

    // determine the final offset from initial (pixel/2) and oversampled (sub-pixel)
    cuSubPixelOffset(offsetInit, offsetZoomIn, offsetFinal,
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
    cuArraysCopyInsert(offsetFinal, offsetImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // snr
    cuArraysCopyInsert(r_snrValue, snrImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // Variance.
    cuArraysCopyInsert(r_covValue, covImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // peak value.
    cuArraysCopyInsert(corrMaxValue, peakValueImage, idxDown_*param->windowsDownInChunk, idxAcross_*param->windowsAcrossInChunk,stream);
    // all done
}



/// constructor
cuAmpcorProcessorOnePass::cuAmpcorProcessorOnePass(cuAmpcorParameter *param_,
    cuArrays<real2_type> *offsetImage_, cuArrays<real_type> *snrImage_, cuArrays<real3_type> *covImage_, cuArrays<real_type> *peakValueImage_,
    stream_t stream_)
    : cuAmpcorProcessor(param_, offsetImage_, snrImage_, covImage_, peakValueImage_, stream_)
{


    c_referenceBatchRaw = new cuArrays<complex_type> (
        param->windowSizeHeightRawEnlarged, param->windowSizeWidthRawEnlarged,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_referenceBatchRaw->allocate();

    c_secondaryBatchRaw = new cuArrays<complex_type> (
        param->searchWindowSizeHeightRaw, param->searchWindowSizeWidthRaw,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchRaw->allocate();

    c_referenceBatchOverSampled = new cuArrays<complex_type> (
            param->windowSizeHeightEnlarged, param->windowSizeWidthEnlarged,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_referenceBatchOverSampled->allocate();

    c_secondaryBatchOverSampled = new cuArrays<complex_type> (
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk, param->windowsAcrossInChunk);
    c_secondaryBatchOverSampled->allocate();

    r_referenceBatchOverSampled = new cuArrays<real_type> (
         param->windowSizeHeight, param->windowSizeWidth,
         param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_referenceBatchOverSampled->allocate();

    r_secondaryBatchOverSampled = new cuArrays<real_type> (
        param->searchWindowSizeHeight, param->searchWindowSizeWidth,
        param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_secondaryBatchOverSampled->allocate();

    referenceBatchOverSampler = new cuOverSamplerC2C(
        c_referenceBatchRaw->height, c_referenceBatchRaw->width, //original size
        c_referenceBatchOverSampled->height, c_referenceBatchOverSampled->width, //oversampled size
        c_referenceBatchRaw->count, stream);

    secondaryBatchOverSampler = new cuOverSamplerC2C(
        c_secondaryBatchRaw->height, c_secondaryBatchRaw->width,
        c_secondaryBatchOverSampled->height, c_secondaryBatchOverSampled->width,
        c_secondaryBatchRaw->count, stream);

    r_corrBatch = new cuArrays<real_type> (
        param->corrWindowSize.x,
        param->corrWindowSize.y,
        param->windowsDownInChunk,
        param->windowsAcrossInChunk);
    r_corrBatch->allocate();

    r_corrBatchZoomIn = new cuArrays<real_type> (
            param->corrZoomInSize.x,
            param->corrZoomInSize.y,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchZoomIn->allocate();

    r_corrBatchZoomInOverSampled = new cuArrays<real_type> (
        param->corrZoomInOversampledSize.x,
        param->corrZoomInOversampledSize.y,
        param->windowsDownInChunk,
        param->windowsAcrossInChunk);
    r_corrBatchZoomInOverSampled->allocate();

    offsetInit = new cuArrays<int2> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetInit->allocate();

    offsetZoomIn = new cuArrays<int2> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetZoomIn->allocate();

    offsetFinal = new cuArrays<real2_type> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    offsetFinal->allocate();

    // the zoom-in correlation surface is extracted around the peak,
    // so the sinc interpolation center needs no extra shift
    maxLocShift = make_int2(0, 0);

    corrMaxValue = new cuArrays<real_type> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    corrMaxValue->allocate();

    r_corrBatchRawZoomIn = new cuArrays<real_type> (
            param->corrRawZoomInHeight,
            param->corrRawZoomInWidth,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    r_corrBatchRawZoomIn->allocate();

    i_corrBatchZoomInValid = new cuArrays<int> (
            param->corrRawZoomInHeight,
            param->corrRawZoomInWidth,
            param->windowsDownInChunk,
            param->windowsAcrossInChunk);
    i_corrBatchZoomInValid->allocate();

    r_corrBatchSum = new cuArrays<real_type> (
                    param->windowsDownInChunk,
                    param->windowsAcrossInChunk);
    r_corrBatchSum->allocate();

    i_corrBatchValidCount = new cuArrays<int> (
                    param->windowsDownInChunk,
                    param->windowsAcrossInChunk);
    i_corrBatchValidCount->allocate();

    i_maxloc = new cuArrays<int2> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    i_maxloc->allocate();

    r_maxval = new cuArrays<real_type> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_maxval->allocate();

    r_snrValue = new cuArrays<real_type> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_snrValue->allocate();

    r_covValue = new cuArrays<real3_type> (param->windowsDownInChunk, param->windowsAcrossInChunk);
    r_covValue->allocate();

    // end of new arrays

    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        corrSincOverSampler = new cuSincOverSamplerR2R(param->corrSurfaceOverSamplingFactor, stream);
    }
    else {
        corrOverSampler= new cuOverSamplerR2R(
            r_corrBatchZoomIn->height, r_corrBatchZoomIn->width,
            r_corrBatchZoomInOverSampled->height, r_corrBatchZoomInOverSampled->width,
            param->windowsDownInChunk*param->windowsAcrossInChunk,
            stream);
    }
    if(param->algorithm == Algorithm::Frequency) {
        cuCorrFreqDomain_OverSampled = new cuFreqCorrelator(
            param->searchWindowSizeHeight, param->searchWindowSizeWidth,
            param->windowsDownInChunk * param->windowsAcrossInChunk,
            stream);
    }


    corrNormalizerOverSampled =
        std::unique_ptr<cuNormalizeProcessor>(newCuNormalizer(
        param->searchWindowSizeHeight,
        param->searchWindowSizeWidth,
        param->windowsDownInChunk * param->windowsAcrossInChunk
        ));


#ifdef CUAMPCOR_DEBUG
    std::cout << "all objects in chunk are created ...\n";
#endif
}

// destructor
cuAmpcorProcessorOnePass::~cuAmpcorProcessorOnePass()
{
    corrNormalizerOverSampled.reset();

    if(param->corrSurfaceOverSamplingMethod == OversamplingMethod::Sinc) {
        delete corrSincOverSampler;
    }
    else {
        delete corrOverSampler;
    }
    if(param->algorithm == Algorithm::Frequency) {
        delete cuCorrFreqDomain_OverSampled;
    }

    delete c_referenceBatchRaw;
    delete c_secondaryBatchRaw;
    delete c_referenceBatchOverSampled;
    delete c_secondaryBatchOverSampled;
    delete r_referenceBatchOverSampled;
    delete r_secondaryBatchOverSampled;
    delete referenceBatchOverSampler;
    delete secondaryBatchOverSampler;

    delete r_corrBatch;
    delete r_corrBatchZoomIn;
    delete r_corrBatchZoomInOverSampled;
    delete offsetInit;
    delete offsetZoomIn;
    delete offsetFinal;
    delete corrMaxValue;

    delete r_corrBatchRawZoomIn;
    delete i_corrBatchZoomInValid;
    delete r_corrBatchSum;
    delete i_corrBatchValidCount;
    delete i_maxloc;
    delete r_maxval;
    delete r_snrValue;
    delete r_covValue;

    // end of deletions

}

// end of file


} // namespace