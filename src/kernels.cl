// vim:filetype=c

void gradient(__read_only image2d_t input, int2 coord, float4* dx, float4* dy)
{
    sampler_t nn_sampler =
        CLK_NORMALIZED_COORDS_FALSE |
        CLK_ADDRESS_CLAMP_TO_EDGE |
        CLK_FILTER_NEAREST;

    float4 l = read_imagef(input, nn_sampler, coord + (int2)(-1, 0));
    float4 r = read_imagef(input, nn_sampler, coord + (int2)( 1, 0));
    *dx = (r - l) * 0.5f;

    float4 b = read_imagef(input, nn_sampler, coord + (int2)(0, -1));
    float4 t = read_imagef(input, nn_sampler, coord + (int2)(0,  1));
    *dy = (t - b) * 0.5f;
}

__kernel void image_slope(
    __read_only image2d_t input,
    __write_only image2d_t output
    )
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
    gradient(input, pixel, &dx, &dy);
    float4 slope = sqrt(dx*dx + dy*dy);

    write_imagef(output, pixel, slope);
}

__kernel void image_aspect(
    __read_only image2d_t input,
    __write_only image2d_t output
    )
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
    gradient(input, pixel, &dx, &dy);
    float4 aspect = atan2(dy, dx);

    write_imagef(output, pixel, aspect);
}
