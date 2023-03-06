#include "pose_estimation_3d2d.h"
#include "common_functions.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: pose_estimation_2d2d path" << endl;
        return 1;
    }
    cout << endl << "Dataset: " << argv[1] << endl;
    string path1 = string(argv[1]) + "/1.png";
    string path2 = string(argv[1]) + "/2.png";
    string path3 = string(argv[1]) + "/1_depth.png";
    string path4 = string(argv[1]) + "/2_depth.png";

    Mat img_1 = imread(path1, IMREAD_COLOR);
    Mat img_2 = imread(path2, IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");


    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "found " << matches.size() << " matches" << endl;


    Mat d1 = imread(path3, IMREAD_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m:matches) {
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        if (d == 0) {
            continue;
        }
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3d(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    cv::Rodrigues(r, R);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    cout << "calling bundle adjustment by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}