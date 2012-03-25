project(pathcl CXX C)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

find_package(PkgConfig REQUIRED)

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

pkg_check_modules(EIGEN eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

add_subdirectory(tools)
add_subdirectory(src)
