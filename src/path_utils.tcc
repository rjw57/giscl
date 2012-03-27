#ifndef INSIDE_GISCL_PATH_UTILS_HPP__
#error "This file must be included only within path_utils.hpp"
#endif

namespace giscl
{

template<typename InputIterator>
float path_cost(
    const raster& gradients,
    InputIterator first_point,
    InputIterator last_point,
    const cl::Context& context,
    cl::Kernel& cost_kernel,
    const cl::CommandQueue& command_queue)
{
    if(last_point == first_point)
        return 0.f;

    InputIterator seg_start(first_point), seg_end(first_point);
    ++seg_end;

    float cost(0.f);
    while(seg_end != last_point)
    {
        cost += line_cost(
            gradients,
            *seg_start,
            *seg_end,
            context,
            cost_kernel,
            command_queue);

        ++seg_start;
        ++seg_end;
    }

    BOOST_ASSERT(seg_end == last_point);

    return cost;
}

}
