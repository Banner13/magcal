/*  ****************************************************************************
 *  magcal: Magnetometer calibration coefficients(form Matlab magcal)
 *  author: jack chatgpt-o4
 *  email:  hihuke@163.com
 *  tip:    Dependency on the Eigen library
 *  more:   https://ww2.mathworks.cn/help/nav/ref/magcal.html?searchHighlight=magcal&s_tid=srchtitle_support_results_1_magcal
 *  ***************************************************************************/

#pragma once


/*  ****************************************************************************
 *  include
 *  ***************************************************************************/
#include <string>
#include <Eigen/Dense>


/*  ****************************************************************************
 *  definition
 *  ***************************************************************************/
struct CalibrationResult {
    Eigen::Matrix3d A;
    Eigen::Vector3d b;
    double magB;
};

/*  ****************************************************************************
 *  phototype
 *  ***************************************************************************/
CalibrationResult magcal(const Eigen::MatrixXd d, const std::string& fitkind = "");



