/**
 * @file cpuUtil.h
 * @brief Various utilities for the CPU backend
 *
 **/

#pragma once

#include "indexing.h"

namespace pycuampcor::cpu {

// compute the next integer in power of 2
inline int nextpower2(int value)
{
    int r=1;
    while (r<value) r<<=1;
    return r;
}

} // namespace
//end of file
