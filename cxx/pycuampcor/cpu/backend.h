/**
 * @file  backend.h
 * @brief Backend hooks used by the shared (backend-agnostic) ampcor code: CPU version
 *
 * The code in common/ is compiled once per backend, in namespace
 * pycuampcor::PYCUAMPCOR_BACKEND (defined by the build system, e.g., cpu).
 * Each backend provides a backend.h with the same interface.
 *
 * For the CPU backend, each worker is an OpenMP thread processing one chunk
 * at a time with its own chunk processor.
 */

#ifndef __PYCUAMPCOR_BACKEND_H
#define __PYCUAMPCOR_BACKEND_H

#include <cstddef>

namespace pycuampcor::cpu {

class cuAmpcorParameter;

/// stream type to pass to the processing kernels (not used by the CPU backend)
struct stream_t {};

/// initialize the compute device; return the device id in use
int backendInit(cuAmpcorParameter *param);
/// number of concurrent chunk processors (cpu threads)
int backendNumWorkers(const cuAmpcorParameter *param);
/// the chunk processor to run the k-th chunk: the current openmp thread
int backendWorkerId(int k, int nWorkers);
/// create/destroy a stream for a worker
inline stream_t backendCreateStream() { return stream_t{}; }
inline void backendDestroyStream(stream_t) {}
/// wait for all workers to finish
inline void backendSynchronize() {}
/// copy a 2D tile from host memory (e.g., mmap) to the backend memory
void backendCopyFromHost2D(void *dst, size_t dpitch, const void *src, size_t spitch,
    size_t widthInBytes, size_t height, stream_t stream);

} // namespace

#endif //__PYCUAMPCOR_BACKEND_H
// end of file
