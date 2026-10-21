/**
 * @file data_types.h
 * @brief The data types, in single precision, or in double precision with CUAMPCOR_DOUBLE
 *
 */

#pragma once

#include <cfloat>
#include <cuda_runtime.h> // for vector types int2, float2, ...
#include "hostDevice.h"

namespace pycuampcor::cuda {

// disable this for single precision version
// #define CUAMPCOR_DOUBLE

#ifdef CUAMPCOR_DOUBLE
using real_type = double;
using complex_type = double2;
using real2_type = double2;
using real3_type = double3;
PYCUAMPCOR_HOST_DEVICE inline real2_type make_real2(real_type x, real_type y) { return make_double2(x, y); }
PYCUAMPCOR_HOST_DEVICE inline real3_type make_real3(real_type x, real_type y, real_type z) { return make_double3(x, y, z); }
PYCUAMPCOR_HOST_DEVICE inline complex_type make_complex_type(real_type x, real_type y) { return make_double2(x, y); }
constexpr real_type EPSILON = DBL_EPSILON;
constexpr real_type REAL_MAX = DBL_MAX;
#else
using real_type = float;
using complex_type = float2;
using real2_type = float2;
using real3_type = float3;
PYCUAMPCOR_HOST_DEVICE inline real2_type make_real2(real_type x, real_type y) { return make_float2(x, y); }
PYCUAMPCOR_HOST_DEVICE inline real3_type make_real3(real_type x, real_type y, real_type z) { return make_float3(x, y, z); }
PYCUAMPCOR_HOST_DEVICE inline complex_type make_complex_type(real_type x, real_type y) { return make_float2(x, y); }
constexpr real_type EPSILON = FLT_EPSILON;
constexpr real_type REAL_MAX = FLT_MAX;
#endif

// the pixel types of the images (single precision)
using image_complex_type = float2;
using image_real_type = float;

} // namespace
//end of file
