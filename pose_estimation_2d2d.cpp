#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include "common_functions.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "pose_estimation_2d2d.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
    main_2d2d(argc, argv);
    return 0;
}

void pose_estimation_2d2d(vector<KeyPoint> keypoint_1,
                          vector<KeyPoint> keypoint_2,
                          vector<DMatch> matches,
                          Mat &R, Mat &t) {
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<Point2f> points1;
    vector<Point2f> points2;

    for (auto & match : matches) {
        points1.push_back(keypoint_1[match.queryIdx].pt);
        points2.push_back(keypoint_2[match.queryIdx].pt);
    }

    Mat fundamental_matrix;

    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "Fundamental Matrix is " << endl << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "Essential Matrix is " << endl << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "Homography Matrix is " << endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}

void triangulation(
        const vector<KeyPoint> &keypoints_1,
        const vector<KeyPoint> &keypoints_2,
        const vector<DMatch> &matches,
        const Mat &R, const Mat &t,
        vector<Point3d> &points) {
    Mat T1 = (Mat_<float>(3, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
            R.at<double>(0 ,0), R.at<double>(0 ,1), R.at<double>(0 ,2), t.at<double>(0 ,0),
            R.at<double>(1 ,0), R.at<double>(1 ,1), R.at<double>(1 ,2), t.at<double>(1 ,0),
            R.at<double>(2 ,0), R.at<double>(2 ,1), R.at<double>(2 ,2), t.at<double>(2 ,0)
            );
    Mat K = (Mat_<double>(3, 3) << 520.9, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for (DMatch m: matches) {
      pts_1.push_back(pixel2cam(keypoints_1[m.queryIdx].pt, K));
      pts_2.push_back(pixel2cam(keypoints_2[m.queryIdx].pt, K));
    }
    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i=0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0);
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1,0),
                x.at<float>(2, 0)
                );
        points.push_back(p);
    }
}

int main_2d2d(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: pose_estimation_2d2d path" << endl;
        return 1;
    }
    cout << endl << "Dataset: " << argv[1] << endl;
    string path1 = string(argv[1]) + "000000.png";
    string path2 = string(argv[1]) + "000001.png";

    Mat img_1 = imread(path1, IMREAD_GRAYSCALE);
    Mat img_2 = imread(path2, IMREAD_GRAYSCALE);
    assert(img_1.data && img_2.data && "Can not load images!");


    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "found " << matches.size() << " matches" << endl;
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    // E = t@R*scale
    Mat t_x =
            (Mat_<double>(3,3) << 0, -t.at<double>(2,0), t.at<double>(1, 0),
                    t.at<double>(2,0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1,0), t.at<double>(0, 0), 0);
    cout << "t@R=" << endl << t_x * R << endl;

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.queryIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    Mat img1_plot;
    cvtColor(img_1.clone(), img1_plot, COLOR_GRAY2BGR);
    Mat img2_plot;
    cvtColor(img_2.clone(), img2_plot, COLOR_GRAY2BGR);
    for (int i = 0; i < matches.size(); i++) {
        cout << "point " << i << endl;
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();
    return 0;
}