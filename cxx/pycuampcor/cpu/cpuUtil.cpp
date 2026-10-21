/**
 * @file cpuUtil.cpp
 * @brief Backend hooks for the shared ampcor code, CPU version
 *
 **/

#include "cpuUtil.h"
#include "logging.h"
#include "cuAmpcorParameter.h"
#include <cstring>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pycuampcor::cpu {

int backendInit(cuAmpcorParameter *param)
{
    logInfo("Using ", backendNumWorkers(param), " CPU thread(s) ...");
    return 0;
}

int backendNumWorkers(const cuAmpcorParameter *param)
{
    if(param->nThreads > 0)
        return param->nThreads;
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void backendDefaultChunkSize(const cuAmpcorParameter *, int &down, int &across)
{
    // each thread processes a chunk, its windows one after another; one window per chunk
    // keeps the work in cache and balances the load among threads
    down = 1;
    across = 1;
}

int backendWorkerId(int, int)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void backendCopyFromHost2D(void *dst, size_t dpitch, const void *src, size_t spitch,
    size_t widthInBytes, size_t height, stream_t)
{
    for(size_t i=0; i<height; i++)
        std::memcpy(static_cast<char *>(dst) + i*dpitch, static_cast<const char *>(src) + i*spitch, widthInBytes);
}

} // namespace
// end of file
