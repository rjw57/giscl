#ifndef GISCL_PATH_UTILS_HPP__
#define GISCL_PATH_UTILS_HPP__

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include "raster.hpp"

namespace giscl
{

/// @brief Return the cost of a line.
///
/// @param gradients A raster giving the gradient vectors for the elevation raster.
/// @param start The start of the line in projection co-ordinates
/// @param end The end of the line in projection co-ordinates
/// @param context
/// @param cost_kernel The "segment_costs" kernel from kernels.cl
/// @param command_queue
float line_cost(
    const raster& gradients,
    const coord_2d& start,
    const coord_2d& end,
    const cl::Context& context,
    cl::Kernel& cost_kernel,
    const cl::CommandQueue& command_queue);

}

#endif // GISCL_PATH_UTILS_HPP__
