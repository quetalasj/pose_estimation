#include "pose_estimation_3d3d.h"
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
#include "vector"

using namespace std;
using namespace cv;

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d , VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}
    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose *>(_vertices[0]);
        _error = _measurement - pose->estimate() * _point;
    }
    virtual void linearizeOplus() override {
        VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
        Sophus::SE3d T = pose->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0,0) = -Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }
    bool read(istream &in) {}
    bool write(ostream & out) const {}
protected:
    Eigen::Vector3d _point;
};


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    cout << endl << "Dataset: " << argv[1] << endl;
    string path1 = string(argv[1]) + "/1.png";
    string path2 = string(argv[1]) + "/2.png";
    string path3 = string(argv[1]) + "/1_depth.png";
    string path4 = string(argv[1]) + "/2_depth.png";

    Mat img_1 = imread(path1, IMREAD_COLOR);
    Mat img_2 = imread(path2, IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "found " << matches.size() << " matches" << endl;

    Mat depth1 = imread(path3, IMREAD_UNCHANGED);
    Mat depth2 = imread(path4, IMREAD_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for (DMatch m:matches) {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0)   // bad depth
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    cout << "calling bundle adjustment"s << endl;

    bundleAdjustment(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1 = "s << pts1[i] << endl;
        cout << "p2 = "s << pts2[i] << endl;
        cout << "(R*p2+t) = "s <<
             R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
             << endl;
        cout << endl;
    }
}


void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
        p1 = Point3f(
            Vec3f(
                    accumulate(pts1.begin(), pts1.end(),Point3f(0,0,0))
            ) / N
            );
    p2 = Point3f(
            Vec3f(
                    accumulate(pts2.begin(), pts2.end(),Point3f(0,0,0))
            ) / N
    );


    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W="s << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    cout << "U="s << U << endl << "V="s << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {
        R_ = -R_;
    }
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}


void bundleAdjustment(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t) {

    typedef g2o::BlockSolverX BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    VertexPose *pose = new VertexPose(); // camera pose
    pose->setId(0);
    pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(pose);

    // edges
    for (size_t i = 0; i < pts1.size(); i++) {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Eigen::Vector3d(
                pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    cout << endl << "after optimization:" << endl;
    cout << "T=\n" << pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = pose->estimate().translation();
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}