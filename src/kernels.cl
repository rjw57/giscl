// vim:filetype=c

inline float lanczos(float2 x, float a)
{
    float len = length(x);
    if((len < 1e-2f) && (len > -1e-2f))
        return 1.f;

    float2 k = (a * sinpi(x) * sinpi(x/a)) / (3.14159f * 3.14159f * x * x);
    return k.x * k.y;
}

inline float4 lanczos_sample(
    __read_only image2d_t input,
    float2 coord)
{
    sampler_t nn_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    float2 centre = round(coord);
    int a = 2;
    float norm = 0.f;
    float4 sample = (float4)(0.f, 0.f, 0.f, 0.f);

    for(int dx=-a; dx<=a; ++dx)
    {
        for(int dy=-a; dy<=a; ++dy)
        {
            float2 pixel_coord = centre + (float2)(dx, dy);
            float2 delta = coord - pixel_coord;

            float k = lanczos(delta, a);

            norm += k;
            sample += k * read_imagef(input, nn_sampler, pixel_coord);
        }
    }

    return sample / norm;
}

struct segment
{
    float2 midpoint_pixel;  // *pixel* co-ordinate of segment midpoint
    float2 extent;          // direction and length in linear units
};

float segment_cost(
    __read_only image2d_t gradient_image,
    __global const struct segment* seg)
{
    float2 midpoint_grad = lanczos_sample(gradient_image, seg->midpoint_pixel).xy;
    float2 direction = normalize(seg->extent);
    float midpoint_slope = dot(direction, midpoint_grad);

    float seg_length = length(seg->extent);
    float vertical_disp = midpoint_slope * seg_length;
    float euc_distance = length((float4)(direction * seg_length, vertical_disp, 0.f));

    return euc_distance;
}

void gradient(__read_only image2d_t input,
              int2 coord,
              float4* dx, float4* dy,
              float2 pixel_to_linear_scale)
{
    sampler_t nn_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    float4 l = read_imagef(input, nn_sampler, coord + (int2)(-1, 0));
    float4 r = read_imagef(input, nn_sampler, coord + (int2)( 1, 0));
    *dx = (r - l) * (0.5f / pixel_to_linear_scale.x);

    float4 b = read_imagef(input, nn_sampler, coord + (int2)(0, -1));
    float4 t = read_imagef(input, nn_sampler, coord + (int2)(0,  1));
    *dy = (t - b) * (0.5f / pixel_to_linear_scale.y);
}

__kernel void segment_costs(
    __read_only image2d_t gradient_image,
    __global struct segment* segments,
    const int n_segments,
    __global float* output_costs)
{
    int idx = get_global_id(0) * get_local_size(0) + get_local_id(0);
    if(idx >= n_segments)
        return;
    output_costs[idx] = segment_cost(gradient_image, segments + idx);
}

__kernel void image_gradient(
    __read_only image2d_t input,
    __write_only image2d_t output,
    float2 pixel_to_linear_scale)
{
    int2 pixel = {
        get_global_id(0) * get_local_size(0) + get_local_id(0),
        get_global_id(1) * get_local_size(1) + get_local_id(1),
    };

    if(pixel.x >= get_image_width(input))
        return;
    if(pixel.y >= get_image_height(input))
        return;

    float4 dx, dy;
    gradient(input, pixel, &dx, &dy, pixel_to_linear_scale);
    write_imagef(output, pixel, (float4)(dx.x, dy.x, 0.f, 0.f));
}

__kernel void image_slope(
    __read_only image2d_t input,
    __write_only image2d_t output,
    float2 pixel_to_linear_scale)
{
    int2 pixel = {
        get_global_id(0) * get_local_size(0) + get_local_id(0),
        get_global_id(1) * get_local_size(1) + get_local_id(1),
    };

    if(pixel.x >= get_image_width(input))
        return;
    if(pixel.y >= get_image_height(input))
        return;

    float4 dx, dy;
    gradient(input, pixel, &dx, &dy, pixel_to_linear_scale);
    float slope = sqrt(dx.x*dx.x + dy.x*dy.x);

    write_imagef(output, pixel, slope);
}

__kernel void image_aspect(
    __read_only image2d_t input,
    __write_only image2d_t output,
    float2 pixel_to_linear_scale)
{
    int2 pixel = {
        get_global_id(0) * get_local_size(0) + get_local_id(0),
        get_global_id(1) * get_local_size(1) + get_local_id(1),
    };

    if(pixel.x >= get_image_width(input))
        return;
    if(pixel.y >= get_image_height(input))
        return;

    float4 dx, dy;
    gradient(input, pixel, &dx, &dy, pixel_to_linear_scale);
    float aspect = atan2(dy.x, dx.x);

    write_imagef(output, pixel, aspect);
}

__kernel void image_hill_shade(
    __read_only image2d_t input,
    __write_only image2d_t output,
    float2 pixel_to_linear_scale)
{
    int2 pixel = {
        get_global_id(0) * get_local_size(0) + get_local_id(0),
        get_global_id(1) * get_local_size(1) + get_local_id(1),
    };

    if(pixel.x >= get_image_width(input))
        return;
    if(pixel.y >= get_image_height(input))
        return;

    float4 dx, dy;
    gradient(input, pixel, &dx, &dy, pixel_to_linear_scale);

    float3 xgrad = (float3)(1.f, 0.f, dx.x);
    float3 ygrad = (float3)(0.f, 1.f, dy.x);
    float4 normal = normalize((float4)(cross(xgrad, ygrad), 0.f));
    float4 light = normalize((float4)(1.f, 1.f, 0.5f, 0.f));

    float shade = clamp(dot(normal, light), 0.f, 1.f);

    write_imagef(output, pixel, shade);
}
