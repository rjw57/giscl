# convert OpenCL sources to C static strings
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.cl.h
    COMMAND bin2c kernels 
        < "${CMAKE_CURRENT_SOURCE_DIR}/kernels.cl"
        > "${CMAKE_CURRENT_BINARY_DIR}/kernels.cl.h"
    DEPENDS kernels.cl
    VERBATIM
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(test
        kernels.cl.h
	main.cpp
        path_utils.cpp
        raster.cpp
)
target_link_libraries(test
	${OPENCL_LIBRARIES}
	${GDAL_LIBRARY}
)
