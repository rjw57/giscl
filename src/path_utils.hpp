#ifndef GISCL_PATH_UTILS_HPP__
#define GISCL_PATH_UTILS_HPP__

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iterator>
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

/// @brief Calculate the cost of an entire path.
///
/// @tparam InputIterator
/// @param gradients
/// @param first_point
/// @param last_point
/// @param context
/// @param cost_kernel The "segment_costs" kernel from kernels.cl
/// @param command_queue
template<typename InputIterator>
float path_cost(
    const raster& gradients,
    InputIterator first_point,
    InputIterator last_point,
    const cl::Context& context,
    cl::Kernel& cost_kernel,
    const cl::CommandQueue& command_queue);

}

#define INSIDE_GISCL_PATH_UTILS_HPP__
#include "path_utils.tcc"
#undef INSIDE_GISCL_PATH_UTILS_HPP__

#endif // GISCL_PATH_UTILS_HPP__
