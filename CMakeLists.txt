cmake_minimum_required(VERSION 3.24)
project(pose_estimation)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include ${G2O_INCLUDE_DIRS})
include_directories(/usr/local/g2o /usr/include/eigen3)

add_executable(pose_estimation
        pose_estimation_2d2d.cpp
        include/pose_estimation_2d2d.h
        include/common_functions.h)

add_executable(pose_estimation_3d2d
        pose_estimation_3d2d.cpp
        include/pose_estimation_3d2d.h
        include/common_functions.h)

target_link_libraries(pose_estimation ${OpenCV_LIBS})

target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS} g2o_core g2o_stuff fmt glog)
