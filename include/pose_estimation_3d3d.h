#pragma once
#include "vector"
#include <numeric>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

#include "common_functions.h"



void pose_estimation_3d3d(const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2, cv::Mat &R, cv::Mat &t);
void bundleAdjustment(
        const vector<Point3f> &points_3d,
        const vector<Point3f> &points_2d,
        Mat &R, Mat &t
);