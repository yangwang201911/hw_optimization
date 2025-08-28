cmake_minimum_required(VERSION 3.0)

set(PRJ_NAME 04_array_max)
project(${PRJ_NAME} LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

include_directories(
	${Utils}
	)

# Original array_max project
file(GLOB ARRAY_MAX_SRC 
	"src/main.cpp")

add_executable(${PRJ_NAME} ${ARRAY_MAX_SRC})
target_link_libraries(${PRJ_NAME} PRIVATE OpenCL::OpenCL)

# DPP OpenCL test project
set(DPP_PRJ_NAME 04_dpp_opencl)
file(GLOB DPP_SRC 
	"src/dpp_main.cpp")

add_executable(${DPP_PRJ_NAME} ${DPP_SRC})
target_link_libraries(${DPP_PRJ_NAME} PRIVATE OpenCL::OpenCL)

# Set C++ standard
set_property(TARGET ${DPP_PRJ_NAME} PROPERTY CXX_STANDARD 11)