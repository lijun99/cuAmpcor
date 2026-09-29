/**
 * @file cuSincOverSampler.cpp
 * @brief Implementation for cuSincOversampler class (CPU backend)
 *
 */

// my declaration
#include "cuSincOverSampler.h"

// dependencies
#include "cuAmpcorUtil.h"
#include <cmath>

namespace pycuampcor::cpu {

/**
 * cuSincOverSamplerR2R constructor
 * @param i_covs oversampling factor
 * @param stream (not used)
 */
cuSincOverSamplerR2R::cuSincOverSamplerR2R(const int i_covs_, stream_t stream_)
 : i_covs(i_covs_)
{
    stream = stream_;
    i_intplength = int(r_relfiltlen/r_beta+0.5f);
    i_filtercoef = i_intplength*i_decfactor;
    r_filter.resize(i_filtercoef+1);
    cuSetupSincKernel();
}

/// destructor
cuSincOverSamplerR2R::~cuSincOverSamplerR2R() = default;

/**
 * Set up the sinc interpolation kernel (coefficient)
 */
void cuSincOverSamplerR2R::cuSetupSincKernel()
{
    // compute some commonly used constants at first
    real_type r_wgthgt =  (1.0f - r_pedestal)/2.0f;
    real_type r_soff = (i_filtercoef-1.0f)/2.0f;
    real_type r_soff_inverse = 1.0f/r_soff;
    real_type r_decfactor_inverse = real_type(1)/i_decfactor;

    for(int i = 0; i <= i_filtercoef; i++) {
        real_type r_wa = i - r_soff;
        real_type r_wgt = (1.0f - r_wgthgt) + r_wgthgt*std::cos(PI*r_wa*r_soff_inverse);
        real_type r_s = r_wa*r_beta*r_decfactor_inverse*PI;
        real_type r_fct;
        if(r_s != 0.0) {
            r_fct = std::sin(r_s)/r_s;
        }
        else {
            r_fct = 1.0f;
        }
        if(i_weight == 1) {
            r_filter[i] = r_fct*r_wgt;
        }
        else {
            r_filter[i] = r_fct;
        }
    }
}

// sinc interpolation for one image, with a given center shift
static void sincInterpolation(const real_type *imageIn, const int inNX, const int inNY,
    real_type *imageOut, const int outNX, const int outNY,
    int2 centerShift, int factor,
    const real_type * r_filter_, const int i_covs_, const int i_decfactor_, const int i_intplength_,
    const int i_startX, const int i_startY, const int i_int_size)
{
    for(int idxX = 0; idxX < i_int_size; idxX++) {
        // determine the output pixel indices
        int outx = idxX + i_startX + centerShift.x*factor;
        if (outx < 0) outx += outNX;
        if (outx >= outNX) outx-=outNX;
        // index in input grids
        real_type r_xout = (real_type)outx/i_covs_;
        // integer part
        int i_xout = int(r_xout);
        // factional part
        real_type r_xfrac = r_xout - i_xout;
        // fractional part in terms of the interpolation kernel grids
        int i_xfrac = int(r_xfrac*i_decfactor_);

        for(int idxY = 0; idxY < i_int_size; idxY++) {
            int outy = idxY + i_startY +  centerShift.y*factor;
            if (outy < 0) outy += outNY;
            if (outy >= outNY) outy-=outNY;
            real_type r_yout = (real_type)outy/i_covs_;
            int i_yout = int(r_yout);
            real_type r_yfrac = r_yout - i_yout;
            int i_yfrac = int(r_yfrac*i_decfactor_);

            real_type intpData = 0.0; // interpolated value
            real_type r_sincwgt = 0.0; // total filter weight

            // iterate over lines of input image
            // i=0 -> -i_intplength/2
            for(int i=0; i < i_intplength_; i++) {
                // find the corresponding pixel in input(unsampled) image
                int inx = i_xout - i + i_intplength_/2;
                if(inx < 0) inx+= inNX;
                if(inx >= inNX) inx-= inNX;

                real_type r_xsinc_coef = r_filter_[i*i_decfactor_+i_xfrac];

                for(int j=0; j< i_intplength_; j++) {
                    // find the corresponding pixel in input(unsampled) image
                    int iny = i_yout - j + i_intplength_/2;
                    if(iny < 0) iny += inNY;
                    if(iny >= inNY) iny -= inNY;

                    real_type r_ysinc_coef = r_filter_[j*i_decfactor_+i_yfrac];
                    // multiply the factors from xy
                    real_type r_sinc_coef = r_xsinc_coef*r_ysinc_coef;
                    // add to total sinc weight
                    r_sincwgt += r_sinc_coef;
                    // multiply by the original signal and add to results
                    intpData += imageIn[inx*inNY+iny]*r_sinc_coef;
                }
            }
            imageOut[outx*outNY + outy] = intpData/r_sincwgt;
        }
    }
}

/**
 * Execute sinc interpolation
 * @param[in] imagesIn input images
 * @param[out] imagesOut output images
 * @param[in] centerShift the shift of interpolation center
 * @param[in] rawOversamplingFactor the multiplier of the centerShift
 * @note rawOversamplingFactor is for the centerShift, not the signal oversampling factor
 */
void cuSincOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    cuArrays<int2> *centerShift, int rawOversamplingFactor)
{
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    // preset all pixels in out image to 0
    imagesOut->setZero(stream);

    for(int idxImage = 0; idxImage < imagesIn->count; idxImage++)
        sincInterpolation(imagesIn->devData + (size_t)idxImage*imagesIn->size, inNX, inNY,
            imagesOut->devData + (size_t)idxImage*imagesOut->size, outNX, outNY,
            centerShift->devData[idxImage], rawOversamplingFactor,
            r_filter.data(), i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size);
}

void cuSincOverSamplerR2R::execute(cuArrays<real_type> *imagesIn, cuArrays<real_type> *imagesOut,
    int2 centerShift, int rawOversamplingFactor)
{
    const int inNX = imagesIn->height;
    const int inNY = imagesIn->width;
    const int outNX = imagesOut->height;
    const int outNY = imagesOut->width;

    // only compute the overampled signals within a window
    const int i_int_range = i_sincwindow * i_covs;
    // set the start pixel, will be shifted by centerShift*oversamplingFactor (from raw image)
    const int i_int_startX = outNX/2 - i_int_range;
    const int i_int_startY = outNY/2 - i_int_range;
    const int i_int_size = 2*i_int_range + 1;
    // preset all pixels in out image to 0
    imagesOut->setZero(stream);

    for(int idxImage = 0; idxImage < imagesIn->count; idxImage++)
        sincInterpolation(imagesIn->devData + (size_t)idxImage*imagesIn->size, inNX, inNY,
            imagesOut->devData + (size_t)idxImage*imagesOut->size, outNX, outNY,
            centerShift, rawOversamplingFactor,
            r_filter.data(), i_covs, i_decfactor, i_intplength, i_int_startX, i_int_startY, i_int_size);
}

} // namespace
// end of file
