/**
 * @file  float2.h
 * @brief Vector types (as in CUDA) and operators on complex (float2/double2) data for the CPU backend
 *
 */

#pragma once

#include <cmath>

namespace pycuampcor::cpu {

/// a pair of ints (x, y), as int2 in CUDA
struct int2 { int x, y; };
/// a triple of ints (x, y, z), as int3 in CUDA
struct int3 { int x, y, z; };
/// a pair of floats (x, y), as float2 in CUDA
struct float2 { float x, y; };
/// a triple of floats (x, y, z), as float3 in CUDA
struct float3 { float x, y, z; };
/// a pair of doubles (x, y), as double2 in CUDA
struct double2 { double x, y; };
/// a triple of doubles (x, y, z), as double3 in CUDA
struct double3 { double x, y, z; };

/// an int2 from its components
inline int2 make_int2(int x, int y) { return {x, y}; }
/// an int3 from its components
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
/// a float2 from its components
inline float2 make_float2(float x, float y) { return {x, y}; }
/// a float3 from its components
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
/// a double2 from its components
inline double2 make_double2(double x, double y) { return {x, y}; }
/// a double3 from its components
inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }

/// define the operators and functions on complex numbers, for T2 = float2, double2 with
/// components (x: real part, y: imaginary part) of type T:
/// - zero(a): set a complex number to zero
/// - -a: the negative of a complex number
/// - conjugate(a): the complex conjugate
/// - a + b, a += b: the sum of two complex numbers; with a real b, added to the real part
/// - a - b, a -= b: the difference of two complex numbers; with a real b, subtracted from the
///   real part
/// - a * b, a *= b: the product of two complex numbers, or a complex number scaled by a real
///   or an int b
/// - complexMul(a, b): the product of two complex numbers, a*b
/// - complexMulConj(a, b): the product of a and the complex conjugate of b, a*conj(b)
/// - a / b, a /= b: a complex number divided by a real b
/// - complexAbs(a): the modulus (absolute value) of a complex number
/// - complexArg(a): the phase (argument) of a complex number, in [-pi, pi]
/// - complexExp(arg): the complex number of modulus 1 and phase arg, exp(i*arg)
#define PYCUAMPCOR_DEFINE_COMPLEX_OPS(T2, T) \
inline void zero(T2 &a) { a.x = 0; a.y = 0; } \
inline T2 operator-(const T2 &a) { return {-a.x, -a.y}; } \
inline T2 conjugate(T2 a) { return {a.x, -a.y}; } \
inline T2 operator+(T2 a, T2 b) { return {a.x + b.x, a.y + b.y}; } \
inline void operator+=(T2 &a, T2 b) { a.x += b.x; a.y += b.y; } \
inline T2 operator+(T2 a, T b) { return {a.x + b, a.y}; } \
inline void operator+=(T2 &a, T b) { a.x += b; } \
inline T2 operator-(T2 a, T2 b) { return {a.x - b.x, a.y - b.y}; } \
inline void operator-=(T2 &a, T2 b) { a.x -= b.x; a.y -= b.y; } \
inline T2 operator-(T2 a, T b) { return {a.x - b, a.y}; } \
inline void operator-=(T2 &a, T b) { a.x -= b; } \
inline T2 operator*(T2 a, T2 b) { return {a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y}; } \
inline void operator*=(T2 &a, T2 b) { a = a*b; } \
inline T2 operator*(T2 a, T b) { return {a.x*b, a.y*b}; } \
inline void operator*=(T2 &a, T b) { a.x *= b; a.y *= b; } \
inline T2 operator*(T2 a, int b) { return {a.x*b, a.y*b}; } \
inline void operator*=(T2 &a, int b) { a.x *= b; a.y *= b; } \
inline T2 complexMul(T2 a, T2 b) { return a*b; } \
inline T2 complexMulConj(T2 a, T2 b) { return {a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y}; } \
inline T2 operator/(T2 a, T b) { return {a.x/b, a.y/b}; } \
inline void operator/=(T2 &a, T b) { a.x /= b; a.y /= b; } \
inline T complexAbs(T2 a) { return std::sqrt(a.x*a.x + a.y*a.y); } \
inline T complexArg(T2 a) { return std::atan2(a.y, a.x); } \
inline T2 complexExp(T arg) { return {std::cos(arg), std::sin(arg)}; }

PYCUAMPCOR_DEFINE_COMPLEX_OPS(float2, float)
PYCUAMPCOR_DEFINE_COMPLEX_OPS(double2, double)

#undef PYCUAMPCOR_DEFINE_COMPLEX_OPS

} // namespace
// end of file
