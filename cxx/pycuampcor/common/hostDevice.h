/**
 * @file  hostDevice.h
 * @brief Annotation of functions callable from host and device (CUDA) code
 */

#pragma once

#ifdef __CUDACC__
#define PYCUAMPCOR_HOST_DEVICE __host__ __device__
#else
#define PYCUAMPCOR_HOST_DEVICE
#endif
//end of file
