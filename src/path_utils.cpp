#include <boost/foreach.hpp>
#include <utility>

#include "path_utils.hpp"
#include "raster.hpp"

namespace giscl
{

struct __attribute__ ((packed)) segment
{
    segment() { midpoint[0] = midpoint[1] = extent[0] = extent[1] = 0.f; }
    segment(const coord_2d& m, const coord_2d& e)
    {
        midpoint[0] = m.get<0>();
        midpoint[1] = m.get<1>();
        extent[0] = e.get<0>();
        extent[1] = e.get<1>();
    }

    float midpoint[2];  // *pixel* co-ordinate of segment midpoint
    float extent[2];    // direction and length in linear units
};

template<typename OutputIterator>
void line_segments(
    const raster& r,
    const coord_2d& start_coord,
    const coord_2d& end_coord,
    OutputIterator output)
{
    using namespace Eigen;

    // convert projection units to pixel
    coord_2d start_pixel(r.proj_to_pixel(start_coord));
    coord_2d end_pixel(r.proj_to_pixel(end_coord));

    // operate in pixels from now on
    Vector2f start(start_pixel.get<0>(), start_pixel.get<1>());
    Vector2f end(end_pixel.get<0>(), end_pixel.get<1>());
    Vector2f delta(end-start);

    float delta_len = delta.norm();
    delta /= delta_len;

    Vector2f last_seg_end(start);
    coord_2d pixel_shape(r.pixel_linear_scale());
    for(float alpha=0.f; alpha < delta_len; alpha += 1.f)
    {
        Vector2f seg_start(last_seg_end);
        Vector2f seg_end(std::min(alpha+1.f, delta_len)*delta + start);
        Vector2f midpoint(0.5f * (seg_end + seg_start));
        Vector2f extent(seg_end - seg_start);

        *output = segment(
            coord_2d(midpoint[0], midpoint[1]),
            coord_2d(extent[0] * pixel_shape.get<0>(), extent[1] * pixel_shape.get<1>()));
        ++output;

        last_seg_end = seg_end;
    }
}

float line_cost(
    const raster& gradients,
    const coord_2d& start,
    const coord_2d& end,
    const cl::Context& context,
    cl::Kernel& cost_kernel,
    const cl::CommandQueue& command_queue)
{
    std::vector<segment> segments;
    line_segments(gradients, start, end, std::back_inserter(segments));

    float crow_dist = 0.f;
    BOOST_FOREACH(const segment& s, segments)
    {
        Eigen::Vector2f seg_ext(s.extent[0], s.extent[1]);
        crow_dist += seg_ext.norm();
    }

    cl::Buffer segment_buffer(
        context,
        CL_MEM_READ_ONLY,
        sizeof(segment) * segments.size());

    cl::Buffer cost_buffer(
        context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * segments.size());

    command_queue.enqueueWriteBuffer(
        segment_buffer,
        CL_TRUE,
        0, sizeof(segment) * segments.size(),
        &(segments.at(0)));

    cost_kernel.setArg(0, *(gradients.data()));
    cost_kernel.setArg(1, segment_buffer);
    cost_kernel.setArg(2, static_cast<int>(segments.size()));
    cost_kernel.setArg(3, cost_buffer);

    command_queue.enqueueNDRangeKernel(
        cost_kernel,
        cl::NullRange,
        cl::NDRange(segments.size()),
        cl::NullRange);
    command_queue.finish();

    std::vector<float> costs(segments.size());
    command_queue.enqueueReadBuffer(
        cost_buffer,
        CL_TRUE,
        0, sizeof(float) * segments.size(),
        &(costs.at(0)));

    float cost_total = 0.f;
    BOOST_FOREACH(const float& c, costs)
    {
        cost_total += c;
    }

    return cost_total;
}

}
