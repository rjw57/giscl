// vim:sw=4:sts=4:et:cino=(0,W4,g0,+0

#define __CL_ENABLE_EXCEPTIONS

#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include <CL/cl.hpp>
#include <cmath>
#include <deque>
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
#include "path_utils.hpp"
#include "raster.hpp"

using namespace giscl;

typedef boost::shared_ptr<cl::Context> context_ptr;
typedef boost::shared_ptr<cl::CommandQueue> command_queue_ptr;
typedef boost::shared_ptr<cl::Image2D> image_2d_ptr;

const coord_2d stonehenge(412500, 142500);
const coord_2d avebury(410500, 169500);

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

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
        bool first(true);
        BOOST_FOREACH(const cl::Device& device, devices)
        {
            if(first)
            {
                std::cout << "  devices: ";
            }
            else
            {
                std::cout << "           ";
                first = false;
            }
            std::cout << device.getInfo<CL_DEVICE_NAME>() << ", "
                      << device.getInfo<CL_DEVICE_VENDOR>() << ", "
                      << device.getInfo<CL_DEVICE_VERSION>() << '\n';
        }
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

    cl::Kernel segment_costs(program, "segment_costs");
    std::cout << "calculating costs..." << std::flush;

    std::deque<coord_2d> path;
    path.push_back(stonehenge);
    path.push_back(avebury);

    float cost = path_cost(
        *gradient,
        path.begin(), path.end(),
        *context,
        segment_costs,
        *command_queue);
    std::cout << " done" << std::endl;

    std::cout << "line cost: " << cost << '\n';

    std::cout << "writing output..." << std::flush;
    write_raster(
        *gradient,
        "output.tiff",
        *command_queue);
    command_queue->finish();
    std::cout << " done" << std::endl;

    return EXIT_SUCCESS;
}
