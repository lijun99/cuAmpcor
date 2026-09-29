/**
 * @file cpuUtil.h
 * @brief Various utilities for the CPU backend
 *
 **/

#ifndef __CPUUTIL_H
#define __CPUUTIL_H

#define IDX2R(i,j,NJ) (((i)*(NJ))+(j))  //row-major order
#define IDX2C(i,j,NI) (((j)*(NI))+(i))  //col-major order

#define IDIVUP(i,j) ((i+j-1)/j)

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) (a > b ? b: a)
#endif

namespace pycuampcor::cpu {

// compute the next integer in power of 2
inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}

} // namespace

#endif //__CPUUTIL_H
//end of file
