/**
 * @file  backend.h
 * @brief Backend hooks used by the shared (backend-agnostic) ampcor code: CUDA version
 *
 * The code in common/ is compiled once per backend. Each backend provides
 * a backend.h with the same interface.
 */

#ifndef __PYCUAMPCOR_BACKEND_H
#define __PYCUAMPCOR_BACKEND_H

#include <cuda_runtime.h>
#include <cstddef>

class cuAmpcorParameter;

/// stream type to pass to the processing kernels
using stream_t = cudaStream_t;

/// initialize the compute device; return the device id in use
int backendInit(cuAmpcorParameter *param);
/// number of concurrent chunk processors (cuda streams)
int backendNumWorkers(const cuAmpcorParameter *param);
/// the chunk processor to run the k-th chunk
inline int backendWorkerId(int k, int nWorkers) { return k % nWorkers; }
/// create/destroy a stream for a worker
stream_t backendCreateStream();
void backendDestroyStream(stream_t stream);
/// wait for all workers to finish
void backendSynchronize();
/// copy a 2D tile from host memory (e.g., mmap) to the backend memory
void backendCopyFromHost2D(void *dst, size_t dpitch, const void *src, size_t spitch,
    size_t widthInBytes, size_t height, stream_t stream);

#endif //__PYCUAMPCOR_BACKEND_H
// end of file
