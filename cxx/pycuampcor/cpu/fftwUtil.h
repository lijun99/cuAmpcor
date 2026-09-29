/*
 * @file  fftwUtil.h
 * @brief Helpers to use fftw (single or double precision) for the CPU backend
 *
 * fftw planning routines are not thread-safe; plans are created/destroyed under a mutex.
 * Plans are executed with the new-array interface on arrays allocated with cuArrays,
 *   which have the same alignment as those used for planning.
 */

#ifndef __FFTWUTIL_H
#define __FFTWUTIL_H

#include "data_types.h"
#include <fftw3.h>
#include <mutex>

namespace pycuampcor::cpu {

#ifdef CUAMPCOR_DOUBLE
    using fftw_plan_type = fftw_plan;
    using fftw_complex_type = fftw_complex;
    #define PYCUAMPCOR_FFTW(name) fftw_ ## name
#else
    using fftw_plan_type = fftwf_plan;
    using fftw_complex_type = fftwf_complex;
    #define PYCUAMPCOR_FFTW(name) fftwf_ ## name
#endif

// fftw planner flag: FFTW_ESTIMATE gives deterministic plans (FFTW_MEASURE may vary between runs)
#define PYCUAMPCOR_FFTW_FLAGS FFTW_ESTIMATE

/// the mutex to guard the fftw planner
std::mutex & fftwPlannerMutex();

/// cast complex arrays to fftw complex type
inline fftw_complex_type * fftwCast(complex_type *data)
{
    return reinterpret_cast<fftw_complex_type *>(data);
}

} // namespace

#endif //__FFTWUTIL_H
// end of file
