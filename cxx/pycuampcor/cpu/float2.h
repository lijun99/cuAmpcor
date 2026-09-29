/*
 * @file  float2.h
 * @brief Vector types (as in CUDA) and operators on complex (float2/double2) data for the CPU backend
 *
 */

#ifndef __FLOAT2_H
#define __FLOAT2_H

#include <cmath>

namespace pycuampcor::cpu {

// vector types with the same layout as those in CUDA
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct double2 { double x, y; };
struct double3 { double x, y, z; };

inline int2 make_int2(int x, int y) { return {x, y}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline double2 make_double2(double x, double y) { return {x, y}; }
inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }

// operators and functions on complex numbers, for T2 = float2, double2 with components of type T
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

#endif //__FLOAT2_H
// end of file
