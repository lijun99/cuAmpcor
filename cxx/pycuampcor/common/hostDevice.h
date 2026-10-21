/**
 * @file  hostDevice.h
 * @brief Annotation of functions callable from host and device (CUDA) code
 */

#ifndef __PYCUAMPCOR_HOSTDEVICE_H
#define __PYCUAMPCOR_HOSTDEVICE_H

#ifdef __CUDACC__
#define PYCUAMPCOR_HOST_DEVICE __host__ __device__
#else
#define PYCUAMPCOR_HOST_DEVICE
#endif

#endif //__PYCUAMPCOR_HOSTDEVICE_H
//end of file
