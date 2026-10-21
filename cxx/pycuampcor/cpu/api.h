/**
 * @file  api.h
 * @brief The public type aliases of the CPU package
 */

#pragma once

#include "forward.h"
// the shared code's aliases: parameter_t, slc_image_t
#include "../common/api.h"

namespace pycuampcor::cpu {
    /// the controller, which runs ampcor with the parameters
    using controller_t = Controller;
    /// the processor of a chunk (the base of the workflows), for the arrays of this package
    using processor_t = pycuampcor::Processor<Arrays>;
}

//end of file
