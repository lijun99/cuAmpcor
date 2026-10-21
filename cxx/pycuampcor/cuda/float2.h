/**
 * @file  float2.h
 * @brief Define operators and functions on float2 (cuComplex) datatype
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace pycuampcor::cuda {

/// set a complex number to zero
inline __host__ __device__ void zero(float2 &a) { a.x = 0.0; a.y = 0.0; }

/// the negative of a complex number
inline __host__ __device__ float2 operator-(float2 &a)
{
    return make_float2(-a.x, -a.y);
}

/// the complex conjugate
inline __host__ __device__ float2 conjugate(float2 a)
{
    return make_float2(a.x, -a.y);
}

/// the sum of two complex numbers
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
/// add a complex number b to a
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
    a.x += b.x;
    a.y += b.y;
}

/// a complex number plus a real b (added to the real part)
inline __host__ __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y);
}
/// add a real b to the real part of a
inline __host__ __device__ void operator+=(float2 &a, float b)
{
    a.x += b;
}

/// the difference of two complex numbers
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
/// subtract a complex number b from a
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
/// a complex number minus a real b (subtracted from the real part)
inline __host__ __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y);
}
/// subtract a real b from the real part of a
inline __host__ __device__ void operator-=(float2 &a, float b)
{
    a.x -= b;
}

/// the product of two complex numbers
inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
/// multiply a by a complex number b
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
    a = make_float2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
/// a complex number scaled by a real b
inline __host__ __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}
/// scale a by a real b
inline __host__ __device__ void operator*=(float2 &a, float b)
{
    a.x *= b;
    a.y *= b;
}
/// a complex number scaled by an int b
inline __host__ __device__ float2 operator*(float2 a, int b)
{
    return make_float2(a.x * b, a.y * b);
}
/// scale a by an int b
inline __host__ __device__ void operator*=(float2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}
/// the product of two complex numbers, a*b
inline __host__ __device__ float2 complexMul(float2 a, float2 b)
{
    return a*b;
}
/// the product of a and the complex conjugate of b, a*conj(b)
inline __host__ __device__ float2 complexMulConj(float2 a, float2 b)
{
    return make_float2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y);
}

/// a complex number divided by a real b
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
/// divide a by a real b
inline __host__ __device__ void operator/=(float2 &a, float b)
{
    a.x /= b;
    a.y /= b;
}

/// the modulus (absolute value) of a complex number
inline __host__ __device__ float complexAbs(float2 a)
{
    return sqrtf(a.x*a.x+a.y*a.y);
}
/// the phase (argument) of a complex number, in [-pi, pi]
inline __host__ __device__ float complexArg(float2 a)
{
    return atan2f(a.y, a.x);
}

/// the complex number of modulus 1 and phase arg, exp(i*arg)
inline __host__ __device__ float2 complexExp(float arg)
{
    return make_float2(cosf(arg), sinf(arg));
}

// double precision
/// set a complex number to zero
inline __host__ __device__ void zero(double2 &a) { a.x = 0.0; a.y = 0.0; }

/// the negative of a complex number
inline __host__ __device__ double2 operator-(double2 &a)
{
    return make_double2(-a.x, -a.y);
}

/// the complex conjugate
inline __host__ __device__ double2 conjugate(double2 a)
{
    return make_double2(a.x, -a.y);
}

/// the sum of two complex numbers
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}
/// add a complex number b to a
inline __host__ __device__ void operator+=(double2 &a, double2 b)
{
    a.x += b.x;
    a.y += b.y;
}

/// a complex number plus a real b (added to the real part)
inline __host__ __device__ double2 operator+(double2 a, double b)
{
    return make_double2(a.x + b, a.y);
}
/// add a real b to the real part of a
inline __host__ __device__ void operator+=(double2 &a, double b)
{
    a.x += b;
}

/// the difference of two complex numbers
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}
/// subtract a complex number b from a
inline __host__ __device__ void operator-=(double2 &a, double2 b)
{
    a.x -= b.x;
    a.y -= b.y;
}
/// a complex number minus a real b (subtracted from the real part)
inline __host__ __device__ double2 operator-(double2 a, double b)
{
    return make_double2(a.x - b, a.y);
}
/// subtract a real b from the real part of a
inline __host__ __device__ void operator-=(double2 &a, double b)
{
    a.x -= b;
}

/// the product of two complex numbers
inline __host__ __device__ double2 operator*(double2 a, double2 b)
{
    return make_double2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
/// multiply a by a complex number b
inline __host__ __device__ void operator*=(double2 &a, double2 b)
{
    a = make_double2(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
}
/// a complex number scaled by a real b
inline __host__ __device__ double2 operator*(double2 a, double b)
{
    return make_double2(a.x * b, a.y * b);
}
/// scale a by a real b
inline __host__ __device__ void operator*=(double2 &a, double b)
{
    a.x *= b;
    a.y *= b;
}
/// a complex number scaled by an int b
inline __host__ __device__ double2 operator*(double2 a, int b)
{
    return make_double2(a.x * b, a.y * b);
}
/// scale a by an int b
inline __host__ __device__ void operator*=(double2 &a, int b)
{
    a.x *= b;
    a.y *= b;
}
/// the product of two complex numbers, a*b
inline __host__ __device__ double2 complexMul(double2 a, double2 b)
{
    return a*b;
}
/// the product of a and the complex conjugate of b, a*conj(b)
inline __host__ __device__ double2 complexMulConj(double2 a, double2 b)
{
    return make_double2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y);
}

/// a complex number divided by a real b
inline __host__ __device__ double2 operator/(double2 a, double b)
{
    return make_double2(a.x / b, a.y / b);
}
/// divide a by a real b
inline __host__ __device__ void operator/=(double2 &a, double b)
{
    a.x /= b;
    a.y /= b;
}

/// the modulus (absolute value) of a complex number
inline __host__ __device__ double complexAbs(double2 a)
{
    return sqrt(a.x*a.x+a.y*a.y);
}
/// the phase (argument) of a complex number, in [-pi, pi]
inline __host__ __device__ double complexArg(double2 a)
{
    return atan2(a.y, a.x);
}

/// the complex number of modulus 1 and phase arg, exp(i*arg)
inline __host__ __device__ double2 complexExp(double arg)
{
    return make_double2(cos(arg), sin(arg));
}

} // namespace
// end of file
