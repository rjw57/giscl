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
#include <utility>
#include <vector>

#include "kernels.cl.h"

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

    image_2d_ptr data() const { return data_; }

    const OGRSpatialReference& spatial_reference() const { return spatial_reference_; }

    /// @brief A matrix mapping from (x,y,1) pixel co-ordinates to homogeneous projection co-ordinates.
    const Eigen::Matrix3f& geo_transform() const { return geo_transform_; }

    size_t width() const { return data_->getImageInfo<CL_IMAGE_WIDTH>(); }

    size_t height() const { return data_->getImageInfo<CL_IMAGE_HEIGHT>(); }

protected:
    image_2d_ptr data_;
    OGRSpatialReference spatial_reference_;
    Eigen::Matrix3f geo_transform_;
};

typedef boost::shared_ptr<raster> raster_ptr;

raster_ptr similar_raster(image_2d_ptr data, const raster& prototype)
{
    char* wkt(NULL);
    prototype.spatial_reference().exportToWkt(&wkt);
    raster_ptr rv(new raster(data, wkt, prototype.geo_transform()));
    OGRFree(wkt);
    return rv;
}

void write_raster(const raster& r, const char* filename, const cl::CommandQueue& command_queue)
{
    GDALDriver* driver(GetGDALDriverManager()->GetDriverByName("GTiff"));
    BOOST_ASSERT(NULL != driver);

    char* options[] = { NULL };
    GDALDataset* dataset(
        driver->Create(filename, r.width(), r.height(), 1, GDT_Float32, options));

    const Eigen::Matrix3f& gt(r.geo_transform());
    double gt_coeff[] = {
        gt(0,2), gt(0,0), gt(0,1),
        gt(1,2), gt(1,0), gt(1,1),
    };
    dataset->SetGeoTransform(gt_coeff);

    char* wkt(NULL);
    r.spatial_reference().exportToWkt(&wkt);
    dataset->SetProjection(wkt);
    OGRFree(wkt);

    size_t pitch(0);
    cl::size_t<3> origin, region;

    origin[0] = origin[1] = origin[2] = 0;
    region[0] = r.width();
    region[1] = r.height();
    region[2] = 1;
    void* image_ptr(command_queue.enqueueMapImage(
            *(r.data()), CL_TRUE, CL_MAP_WRITE,
            origin, region,
            &pitch, NULL));

    dataset->RasterIO(
        GF_Write,
        0, 0,
        r.width(), r.height(),
        image_ptr,
        r.width(), r.height(),
        GDT_Float32,
        1, NULL,
        r.data()->getImageInfo<CL_IMAGE_ELEMENT_SIZE>(),
        pitch,
        0);

    command_queue.enqueueUnmapMemObject(*(r.data()), image_ptr);

    GDALClose(dataset);
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

    boost::shared_ptr<cl::Image2D> image(new cl::Image2D(
            context,
            CL_MEM_READ_ONLY,
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
        gt_coeffs[4], gt_coeffs[5], gt_coeffs[3],
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
            std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << '\n';
        }
        return EXIT_FAILURE;
    }
    std::cout << " done" << std::endl;

    std::cout << "loading elevation..." << std::flush;
    raster_ptr raster(open_raster(argv[1], *context, *command_queue));
    std::cout << " done" << std::endl;

    boost::shared_ptr<cl::Image2D> aspect_image(new cl::Image2D(
            *context,
            CL_MEM_READ_WRITE,
            cl::ImageFormat(CL_INTENSITY, CL_FLOAT),
            raster->width(), raster->height()));

    cl::Kernel aspect_kernel(program, "image_aspect");
    aspect_kernel.setArg(0, *(raster->data()));
    aspect_kernel.setArg(1, *(aspect_image));

    std::cout << "calculating aspect..." << std::flush;
    command_queue->enqueueNDRangeKernel(
        aspect_kernel,
        cl::NDRange(0,0),
        cl::NDRange(raster->width(), raster->height()),
        cl::NullRange);
    command_queue->finish();
    std::cout << " done" << std::endl;

    std::cout << "writing output..." << std::flush;
    write_raster(
        *similar_raster(aspect_image, *raster),
        "output.tiff",
        *command_queue);
    command_queue->finish();
    std::cout << " done" << std::endl;

    return EXIT_SUCCESS;
}
