/**
 * @file  api.h
 * @brief The public type aliases of the CUDA package
 */

#pragma once

#include "forward.h"
// the shared code's aliases: parameter_t, slc_image_t
#include "../common/api.h"

namespace pycuampcor::cuda {
    /// the controller, which runs ampcor with the parameters
    using controller_t = Controller;
    /// the processor of a chunk (the base of the workflows), for the arrays of this package
    using processor_t = pycuampcor::Processor<cuArrays>;
}

//end of file
