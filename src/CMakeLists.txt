add_executable(test
	main.cpp
)
target_link_libraries(test
	${OPENCL_LIBRARIES}
	${GDAL_LIBRARY}
)
