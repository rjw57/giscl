// vim:filetype=c

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
