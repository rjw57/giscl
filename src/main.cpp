// vim:sw=4:sts=4:et:cino=(0,W4,g0,+0

#define __CL_ENABLE_EXCEPTIONS

#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include <CL/cl.hpp>
#include <Eigen/Dense>
#include <error.h>
#include <gdal_priv.h>
#include <iostream>
#include <limits>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

typedef boost::shared_ptr<cl::Context> context_ptr;
typedef boost::shared_ptr<cl::CommandQueue> command_queue_ptr;
typedef boost::shared_ptr<cl::Image2D> image_2d_ptr;
typedef boost::shared_ptr<GDALDataset> dataset_ptr;

class raster : boost::noncopyable
{
public:
    raster(image_2d_ptr data,
           const char* projection_wkt,
           const Eigen::Matrix3f& geo_transform)
        : data_(data)
        , spatial_reference_(projection_wkt)
        , geo_transform_(geo_transform)
    { }

    const OGRSpatialReference& spatial_reference() const { return spatial_reference_; }

    /// @brief A matrix mapping from (x,y,1) pixel co-ordinates to homogeneous projection co-ordinates.
    const Eigen::Matrix3f& geo_transform() const { return geo_transform_; }

protected:
    image_2d_ptr data_;
    OGRSpatialReference spatial_reference_;
    Eigen::Matrix3f geo_transform_;
};

typedef boost::shared_ptr<raster> raster_ptr;

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

raster_ptr open_raster(
    const char* filename,
    const cl::Context& context,
    const cl::CommandQueue& command_queue) throw (std::runtime_error)
{
    dataset_ptr dataset(reinterpret_cast<GDALDataset*>(GDALOpen(filename, GA_ReadOnly)));
    if(!dataset)
    {
        throw std::runtime_error("cannot open dataset");
    }

    std::cout 
        << "raster size: "
        << "(" << dataset->GetRasterXSize()
        << "," << dataset->GetRasterYSize()
        << ")\n";

    boost::shared_ptr<cl::Image2D> image(new cl::Image2D(
            context,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
            cl::ImageFormat(CL_INTENSITY, CL_FLOAT),
            dataset->GetRasterXSize(), dataset->GetRasterYSize()));

    size_t pitch(0);
    cl::size_t<3> origin, region;

    origin[0] = origin[1] = origin[2] = 0;
    region[0] = dataset->GetRasterXSize();
    region[1] = dataset->GetRasterYSize();
    region[2] = 1;

    void* image_ptr(command_queue.enqueueMapImage(
            *image, CL_TRUE, CL_MAP_WRITE,
            origin, region,
            &pitch, NULL));

    dataset->RasterIO(
        GF_Read,
        0, 0,
        dataset->GetRasterXSize(), dataset->GetRasterYSize(),
        image_ptr,
        dataset->GetRasterXSize(), dataset->GetRasterYSize(),
        GDT_Float32,
        1, NULL,
        image->getImageInfo<CL_IMAGE_ELEMENT_SIZE>(),
        pitch,
        0);

    command_queue.enqueueUnmapMemObject(*image, image_ptr);

    double gt_coeffs[6];
    dataset->GetGeoTransform(gt_coeffs);
    Eigen::Matrix3f geo_transform;
    geo_transform <<
        gt_coeffs[1], gt_coeffs[2], gt_coeffs[0],
        gt_coeffs[4], gt_coeffs[5], gt_coeffs[1],
        0, 0, 1;

    return raster_ptr(new raster(image, dataset->GetProjectionRef(), geo_transform));
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

    raster_ptr raster(open_raster(argv[1], *context, *command_queue));

    return EXIT_SUCCESS;
}
