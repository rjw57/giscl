#ifndef GISCL_RASTER_HPP__
#define GISCL_RASTER_HPP__

#define __CL_ENABLE_EXCEPTIONS

#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/utility.hpp>
#include <CL/cl.hpp>
#include <Eigen/Dense>
#include <ogr_spatialref.h>
#include <stdexcept>

namespace giscl
{

typedef boost::shared_ptr<cl::Image2D> image_2d_ptr;
typedef boost::tuple<float, float> coord_2d;

class raster : boost::noncopyable
{
public:
    raster(image_2d_ptr data,
           const char* projection_wkt,
           const Eigen::Matrix3f& geo_transform)
        : data_(data)
        , spatial_reference_(projection_wkt)
        , geo_transform_(geo_transform)
        , inv_geo_transform_(geo_transform.inverse())
    { }

    image_2d_ptr data() const { return data_; }

    const OGRSpatialReference& spatial_reference() const { return spatial_reference_; }

    /// @brief A matrix mapping from (x,y,1) pixel co-ordinates to homogeneous projection co-ordinates.
    const Eigen::Matrix3f& geo_transform() const { return geo_transform_; }

    size_t width() const { return data_->getImageInfo<CL_IMAGE_WIDTH>(); }

    size_t height() const { return data_->getImageInfo<CL_IMAGE_HEIGHT>(); }

    size_t bands() const { return data_->getImageInfo<CL_IMAGE_ELEMENT_SIZE>() >> 2; }

    /// @brief The size of one pixel in linear scale units.
    boost::tuple<float, float> pixel_linear_scale() const;

    /// @brief The size of one pixel in projection units.
    boost::tuple<float, float> pixel_proj() const;

    coord_2d pixel_to_proj(const coord_2d& p) const;

    coord_2d proj_to_pixel(const coord_2d& p) const;

protected:
    image_2d_ptr data_;
    OGRSpatialReference spatial_reference_;
    Eigen::Matrix3f geo_transform_;
    Eigen::Matrix3f inv_geo_transform_;
};

typedef boost::shared_ptr<raster> raster_ptr;

raster_ptr similar_raster(image_2d_ptr data, const raster& prototype);

void write_raster(
    const raster& r,
    const char* filename,
    const cl::CommandQueue& command_queue);

raster_ptr open_raster(
    const char* filename,
    const cl::Context& context,
    const cl::CommandQueue& command_queue) throw (std::runtime_error);

}

#endif // GISCL_RASTER_HPP__
