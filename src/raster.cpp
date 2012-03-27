// vim:sw=4:sts=4:et:cino=(0,W4,g0,+0

#define __CL_ENABLE_EXCEPTIONS

#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <CL/cl.hpp>
#include <cmath>
#include <Eigen/Dense>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <stdlib.h>

#include "raster.hpp"

namespace giscl
{

boost::tuple<float, float> raster::pixel_linear_scale() const
{
    float linear_units = spatial_reference_.GetLinearUnits();
    float pixel_width = geo_transform_(0,0) * linear_units;
    float pixel_height = geo_transform_(1,1) * linear_units;
    return boost::make_tuple(pixel_width, pixel_height);
}

coord_2d raster::pixel_to_proj(const coord_2d& p) const
{
    Eigen::Vector3f pv(p.get<0>(), p.get<1>(), 1.f);
    Eigen::Vector3f tv(geo_transform_ * pv);
    return coord_2d(tv[0]/tv[2], tv[1]/tv[2]);
}

coord_2d raster::proj_to_pixel(const coord_2d& p) const
{
    Eigen::Vector3f pv(p.get<0>(), p.get<1>(), 1.f);
    Eigen::Vector3f tv(inv_geo_transform_ * pv);
    return coord_2d(tv[0]/tv[2], tv[1]/tv[2]);
}

raster_ptr similar_raster(image_2d_ptr data, const raster& prototype)
{
    char* wkt(NULL);
    prototype.spatial_reference().exportToWkt(&wkt);
    raster_ptr rv(new raster(data, wkt, prototype.geo_transform()));
    OGRFree(wkt);
    return rv;
}

raster_ptr open_raster(
    const char* filename,
    const cl::Context& context,
    const cl::CommandQueue& command_queue) throw (std::runtime_error)
{
    typedef boost::shared_ptr<GDALDataset> dataset_ptr;

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

    // replace None with nan
    size_t pix_size(image->getImageInfo<CL_IMAGE_ELEMENT_SIZE>());
    double nodata = dataset->GetRasterBand(1)->GetNoDataValue();
    for(off_t row=0; row<dataset->GetRasterYSize(); ++row)
    {
        off_t pix_offset(row * pitch);
        for(off_t col=0; col<dataset->GetRasterXSize(); ++col, pix_offset += pix_size)
        {
            float* pix = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(image_ptr) + pix_offset);
            if(*pix == nodata)
                *pix = nanf("");
        }
    }

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

void write_raster(const raster& r, const char* filename, const cl::CommandQueue& command_queue)
{
    GDALDriver* driver(GetGDALDriverManager()->GetDriverByName("GTiff"));
    BOOST_ASSERT(NULL != driver);

    char* options[] = { NULL };
    GDALDataset* dataset(
        driver->Create(filename, r.width(), r.height(), r.bands(), GDT_Float32, options));

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
        r.bands(), NULL,
        r.data()->getImageInfo<CL_IMAGE_ELEMENT_SIZE>(),
        pitch,
        4);

    command_queue.enqueueUnmapMemObject(*(r.data()), image_ptr);

    GDALClose(dataset);
}

}
