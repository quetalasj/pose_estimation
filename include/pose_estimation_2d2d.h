#pragma once

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img1, const Mat &img_2,
        vector<KeyPoint> &keypoints_1,
        vector<KeyPoint> &keypoints_2,
        vector<DMatch> &matches);

void pose_estimation_2d2d(
        vector<KeyPoint> keypoints_1,
        vector<KeyPoint> keypoints_2,
        vector<DMatch> matches,
        Mat &R, Mat &t);

void triangulation(
        const vector<KeyPoint> &keypoints_1,
        const vector<KeyPoint> &keypoints_2,
        const vector<DMatch> &matches,
        const Mat &R, const Mat &t,
        vector<Point3d> &points
);

int main_2d2d(int argc, char **argv);

inline Scalar get_color(float depth) {
    float up_th = 0.5, low_th=0.1, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

