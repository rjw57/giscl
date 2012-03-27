// vim:sw=4:sts=4:et:cino=(0,W4,g0,+0

#define __CL_ENABLE_EXCEPTIONS

#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include <CL/cl.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <error.h>
#include <gdal_priv.h>
#include <iostream>
#include <limits>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <stdlib.h>
#include <utility>
#include <vector>

#include "kernels.cl.h"
#include "raster.hpp"

using namespace giscl;

typedef boost::shared_ptr<cl::Context> context_ptr;
typedef boost::shared_ptr<cl::CommandQueue> command_queue_ptr;
typedef boost::shared_ptr<cl::Image2D> image_2d_ptr;

const coord_2d stonehenge(412500, 142500);
const coord_2d avebury(410500, 169500);

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

raster_ptr elevation_gradient(const raster& elevation,
                              const cl::Context& context,
                              const cl::Program& program,
                              const cl::CommandQueue& command_queue)
{
    image_2d_ptr gradient_image(new cl::Image2D(
            context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_RG, CL_FLOAT),
            elevation.width(), elevation.height()));

    cl::Kernel gradient_kernel(program, "image_gradient");
    gradient_kernel.setArg(0, *(elevation.data()));
    gradient_kernel.setArg(1, *(gradient_image));
    boost::tuple<float, float> pixel_linear_scale(elevation.pixel_linear_scale());
    gradient_kernel.setArg(2, pixel_linear_scale);

    std::cout << "calculating gradient..." << std::flush;
    command_queue.enqueueNDRangeKernel(
        gradient_kernel,
        cl::NDRange(0,0),
        cl::NDRange(elevation.width(), elevation.height()),
        cl::NullRange);
    command_queue.finish();
    std::cout << " done" << std::endl;

    return similar_raster(gradient_image, elevation);
}

context_ptr create_opencl_context() throw (std::runtime_error)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if(platforms.empty())
    {
        throw std::runtime_error("no OpenCL platforms found");
    }

    std::cout << "Platforms:\n";
    BOOST_FOREACH(const cl::Platform& platform, platforms)
    {
        std::cout << "     name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        std::cout << "   vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << '\n';
        std::cout << "  version: " << platform.getInfo<CL_PLATFORM_VERSION>() << '\n';
    }

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
    if(devices.empty())
    {
        throw std::runtime_error("no OpenCL devices available");
    }

    context_ptr context(new cl::Context(devices));
    std::cout << "Devices to use:\n";
    BOOST_FOREACH(const cl::Device& device, context->getInfo<CL_CONTEXT_DEVICES>())
    {
        std::cout << "     name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        std::cout << "   vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << '\n';
        std::cout << "  version: " << device.getInfo<CL_DEVICE_VERSION>() << '\n';
    }

    return context;
}

int main(int argc, char** argv)
{
    GDALAllRegister();

    if(argc < 2)
    {
        error(EXIT_FAILURE, 0, "no dataset specified");
    }

    context_ptr context(create_opencl_context());
    cl::Device device(context->getInfo<CL_CONTEXT_DEVICES>()[0]);
    command_queue_ptr command_queue(new cl::CommandQueue(*context, device));

    std::cout << "compiling kernels..." << std::flush;
    std::vector<std::pair<const char*, size_t> > sources;
    sources.push_back(std::make_pair(kernels, sizeof(kernels)-1));
    cl::Program program(*context, sources);

    try
    {
        program.build(context->getInfo<CL_CONTEXT_DEVICES>());
    }
    catch(cl::Error& err)
    {
        std::cerr << "build failed:\n";
        BOOST_FOREACH(const cl::Device& device, context->getInfo<CL_CONTEXT_DEVICES>())
        {
            std::cerr << "build log for " << device.getInfo<CL_DEVICE_NAME>() << ":\n";
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        }
        error(EXIT_FAILURE, 0, "build failed");
    }
    std::cout << " done" << std::endl;

    std::cout << "loading elevation..." << std::flush;
    raster_ptr raster(open_raster(argv[1], *context, *command_queue));
    std::cout << " done" << std::endl;

    raster_ptr gradient(elevation_gradient(*raster, *context, program, *command_queue));

    std::cout << "writing output..." << std::flush;
    write_raster(
        *gradient,
        "output.tiff",
        *command_queue);
    command_queue->finish();
    std::cout << " done" << std::endl;

    std::vector<segment> segments;
    line_segments(*raster, stonehenge, avebury, std::back_inserter(segments));

    float crow_dist = 0.f;
    BOOST_FOREACH(const segment& s, segments)
    {
        Eigen::Vector2f seg_ext(s.extent[0], s.extent[1]);
        crow_dist += seg_ext.norm();
    }
    std::cout << "Crow distance: " << crow_dist << '\n';

    cl::Buffer segment_buffer(
        *context,
        CL_MEM_READ_ONLY,
        sizeof(segment) * segments.size());

    cl::Buffer cost_buffer(
        *context,
        CL_MEM_WRITE_ONLY,
        sizeof(float) * segments.size());

    command_queue->enqueueWriteBuffer(
        segment_buffer,
        CL_TRUE,
        0, sizeof(segment) * segments.size(),
        &(segments.at(0)));

    cl::Kernel cost_kernel(program, "segment_costs");
    cost_kernel.setArg(0, *(gradient->data()));
    cost_kernel.setArg(1, segment_buffer);
    cost_kernel.setArg(2, static_cast<int>(segments.size()));
    cost_kernel.setArg(3, cost_buffer);

    std::cout << "calculating costs..." << std::flush;
    command_queue->enqueueNDRangeKernel(
        cost_kernel,
        cl::NullRange,
        cl::NDRange(segments.size()),
        cl::NullRange);
    command_queue->finish();
    std::cout << " done" << std::endl;

    std::vector<float> costs(segments.size());
    command_queue->enqueueReadBuffer(
        cost_buffer,
        CL_TRUE,
        0, sizeof(float) * segments.size(),
        &(costs.at(0)));

    std::cout << costs.size() << " vs. " << segments.size() << '\n';

    float cost_total = 0.f;
    BOOST_FOREACH(const float& c, costs)
    {
        cost_total += c;
    }
    std::cout << "Total cost: " << cost_total << '\n';

    return EXIT_SUCCESS;
}
