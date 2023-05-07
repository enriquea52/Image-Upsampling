cmake_minimum_required(VERSION 3.9)

project( stereo_estimation LANGUAGES C CXX CUDA )

###############
# packages #
###############

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

ADD_LIBRARY(cudaLibs src/kernels.cu)

###############
# executables #
###############

set_source_files_properties(Advanced_Filtering PROPERTIES LANGUAGE "CUDA")
add_executable( Advanced_Filtering src/main.cu )
target_link_libraries( Advanced_Filtering ${OpenCV_LIBS} cudaLibs)
set_target_properties( Advanced_Filtering PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(pointcloud PROPERTIES LANGUAGE "CUDA")
add_executable( pointcloud src/points.cu )
target_link_libraries( pointcloud ${OpenCV_LIBS} cudaLibs)
set_target_properties( pointcloud PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)


set_source_files_properties(pointcloud PROPERTIES LANGUAGE "CUDA")
add_executable( gui_app src/gui.cu )
target_link_libraries( gui_app ${OpenCV_LIBS} cudaLibs)
set_target_properties( gui_app PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)