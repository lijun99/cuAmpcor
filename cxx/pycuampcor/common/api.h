/**
 * @file  api.h
 * @brief The public type aliases of the package
 */

#pragma once

#include "forward.h"

namespace pycuampcor::PYCUAMPCOR_BACKEND {
    // the parameters
    using parameter_t = cuAmpcorParameter;
    // the controller, which runs ampcor with the parameters
    using controller_t = cuAmpcorController;
    // batches of 2D arrays, in the backend (device or work) memory
    template <typename T>
    using arrays_t = cuArrays<T>;
    // images
    using slc_image_t = SlcImage;
}

//end of file
