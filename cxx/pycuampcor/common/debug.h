/**
 * @file debug.h
 * @brief Define flags to control the debugging
 *
 * CUAMPCOR_DEBUG is used to output debugging information and intermediate results,
 *    disabled when NDEBUG macro is defined.
 * CUDA_ERROR_CHECK is always enabled, to check CUDA routine errors
 *
 */

#pragma once

#ifndef NDEBUG
#define CUAMPCOR_DEBUG
#endif //NDEBUG

#define CUDA_ERROR_CHECK
//end of file
