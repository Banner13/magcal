/*  ****************************************************************************
 *  magcal: Magnetometer calibration coefficients(form Matlab magcal)
 *  author: jack
 *  email:  hihuke@163.com
 *  ***************************************************************************/


/*  ****************************************************************************
 *  include
 *  ***************************************************************************/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include "magcal.h"

// test dataset
#ifdef MAG_TEST
#include "testdata.h"
#endif

/*  ****************************************************************************
 *  definition
 *  ***************************************************************************/

using namespace std;
using namespace Eigen;


struct EllipsoidResult {
    Eigen::Matrix3d Winv;
    Eigen::Vector3d V;
    double B;
    double er;
    bool ispd;
};

/*  ****************************************************************************
 *  code
 *  ***************************************************************************/

static Eigen::VectorXd smallestEigenVector(const Eigen::MatrixXd& d)
{
    // 计算 d 的奇异值分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(d, Eigen::ComputeFullV);
    // 返回 V 矩阵的最后一列（最小特征值对应的特征向量）
    Eigen::VectorXd v = svd.matrixV().col(svd.matrixV().cols() - 1);
    return v;
}

static Eigen::VectorXd residual(const Eigen::Matrix3d& Winv, 
                         const Eigen::Vector3d& V, 
                         const double& B, 
                         const Eigen::MatrixXd& data)
{
    // Ensure that data is N x 3
    assert(data.cols() == 3 && "data must have 3 columns");

    // Calculate the sphere point: (Winv * (data' - V))'
    Eigen::MatrixXd centered = data.transpose().colwise() - V; // 3 x N
    Eigen::MatrixXd spherept = (Winv * centered).transpose(); // N x 3

    // Compute the squared norm for each row
    Eigen::VectorXd radsq = spherept.rowwise().squaredNorm(); // N x 1

    // Compute the residual (radsq - B^2)
    Eigen::VectorXd r = radsq.array() - B * B;
    
    return r;
}


static EllipsoidResult correctEllipsoid4(const Eigen::VectorXd& x, 
                                    const Eigen::VectorXd& y, 
                                    const Eigen::VectorXd& z) 
{
    EllipsoidResult result;

    Eigen::Matrix3d& Winv = result.Winv;
    Eigen::Vector3d& V = result.V;
    double& B = result.B;
    double& er = result.er;
    bool& ispd = result.ispd;

    // 计算 bv = x.^2 + y.^2 + z.^2
    Eigen::VectorXd bv = x.array().square() + y.array().square() + z.array().square();

    // 构造矩阵 A
    Eigen::MatrixXd A(x.size(), 4);
    A << x, y, z, Eigen::VectorXd::Ones(x.size());

    // 解线性方程 A * soln = bv
    Eigen::VectorXd soln = A.colPivHouseholderQr().solve(bv);

    // 计算 Winv, V 和 B

    Winv = Eigen::Matrix3d::Identity();
    V = 0.5 * soln.head(3);
    B = std::sqrt(soln(3) + V.squaredNorm());

    // 如果需要计算误差和 ispd
    Eigen::VectorXd res = A * soln - bv;
    er = 1.0 / (2.0 * B * B) * std::sqrt(res.squaredNorm() / x.size());
    ispd = true;

    return result;
}

static EllipsoidResult correctEllipsoid7(const Eigen::VectorXd& x, 
                                        const Eigen::VectorXd& y, 
                                        const Eigen::VectorXd& z)
{
    EllipsoidResult result;

    Eigen::Matrix3d& Winv = result.Winv;
    Eigen::Vector3d& V = result.V;
    double& B = result.B;
    double& er = result.er;
    bool& ispd = result.ispd;

    int n = x.size();
    Eigen::MatrixXd d(n, 7);
    d.col(0) = x.array().square();
    d.col(1) = y.array().square();
    d.col(2) = z.array().square();
    d.col(3) = x;
    d.col(4) = y;
    d.col(5) = z;
    d.col(6) = Eigen::VectorXd::Ones(n);

    Eigen::VectorXd beta = smallestEigenVector(d);
   
    Eigen::Matrix3d A;
    A << beta(0), 0, 0,
         0, beta(1), 0,
         0, 0, beta(2);

    double dA = beta(0) * beta(1) * beta(2);

    if (dA < 0) {
        A = -A;
        beta = -beta;
        dA = -dA;
    }

    V = -0.5 * beta.segment(3, 3).array() / beta.head(3).array();

    B = std::sqrt(std::abs(A(0, 0) * V(0) * V(0) +
                          A(1, 1) * V(1) * V(1) +
                          A(2, 2) * V(2) * V(2) -
                          beta(6)));

    double det3root = std::pow(dA, 1.0 / 3.0);
    double det6root = std::sqrt(det3root);

    // 计算 scaled_A = A / det3root
    Eigen::Matrix3d scaled_A = A / det3root;

    // 计算 scaled_A 的平方根
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scaled_A);
    Eigen::Matrix3d tD = solver.eigenvalues().asDiagonal(); // 特征值对角矩阵
    Eigen::Matrix3d tV = solver.eigenvectors();            // 特征向量矩阵

    // 计算矩阵平方根: sqrtm(scaled_A) = V * sqrt(D) * V^T
    Eigen::Matrix3d sqrtD = tD.unaryExpr([](double x) { return std::sqrt(x); });
    
    Winv = tV * sqrtD * tV.transpose();
    B /= det6root;

    Eigen::MatrixXd data(n, 3);
    data << x, y, z;
    Eigen::VectorXd res = residual(Winv, V, B, data);
    double sum_res_squared = res.squaredNorm();
    er = (1.0 / (2.0 * B * B)) * std::sqrt(sum_res_squared / n);
    Eigen::LLT<Eigen::Matrix3d> chol(A);
    ispd = (chol.info() == Eigen::Success);

    return result;
}

static EllipsoidResult correctEllipsoid10(const Eigen::VectorXd& x, 
                                        const Eigen::VectorXd& y, 
                                        const Eigen::VectorXd& z) 
{
    EllipsoidResult result;

    Eigen::Matrix3d& Winv = result.Winv;
    Eigen::Vector3d& V = result.V;
    double& B = result.B;
    double& er = result.er;
    bool& ispd = result.ispd;

    int n = x.size();

    // Construct the matrix 'd'
    Eigen::MatrixXd d(n, 10);
    d.col(0) = x.array().square();
    d.col(1) = 2 * x.array() * y.array();
    d.col(2) = 2 * x.array() * z.array();
    d.col(3) = y.array().square();
    d.col(4) = 2 * y.array() * z.array();
    d.col(5) = z.array().square();
    d.col(6) = x;
    d.col(7) = y;
    d.col(8) = z;
    d.col(9) = Eigen::VectorXd::Ones(n);

    Eigen::VectorXd beta = smallestEigenVector(d);
   
    Eigen::Matrix3d A;
    A << beta(0), beta(1), beta(2),
         beta(1), beta(3), beta(4),
         beta(2), beta(4), beta(5);

    double dA = A.determinant();

    if (dA < 0) {
        A = -A;
        beta = -beta;
        dA = -dA;
    }

    V = -0.5 * A.inverse() * beta.segment(6, 3); // hard iron offset

    B = std::sqrt(std::abs(
        A(0,0) * V(0) * V(0) + 2 * A(1,0) * V(1) * V(0) + 2 * A(2,0) * V(2) * V(0) +
        A(1,1) * V(1) * V(1) + 2 * A(2,1) * V(2) * V(1) + A(2,2) * V(2) * V(2) -
        beta(9)
    ));

    double det3root = std::cbrt(dA);
    double det6root = std::sqrt(det3root);

    // 计算 scaled_A = A / det3root
    Eigen::Matrix3d scaled_A = A / det3root;

    // 计算 scaled_A 的平方根
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scaled_A);
    Eigen::Matrix3d tD = solver.eigenvalues().asDiagonal(); // 特征值对角矩阵
    Eigen::Matrix3d tV = solver.eigenvectors();            // 特征向量矩阵

    // 计算矩阵平方根: sqrtm(scaled_A) = V * sqrt(D) * V^T
    Eigen::Matrix3d sqrtD = tD.unaryExpr([](double x) { return std::sqrt(x); });
    
    Winv = tV * sqrtD * tV.transpose();
    B /= det6root;

    Eigen::MatrixXd data(n, 3);
    data << x, y, z;
    Eigen::VectorXd res = residual(Winv, V, B, data);
    double sum_res_squared = res.squaredNorm();
    er = (1.0 / (2.0 * B * B)) * std::sqrt(sum_res_squared / n);
    Eigen::LLT<Eigen::Matrix3d> chol(A);
    ispd = (chol.info() == Eigen::Success);

    return result;
}

static CalibrationResult bestfit(const Eigen::VectorXd& x, 
                            const Eigen::VectorXd& y, 
                            const Eigen::VectorXd& z)
{
    CalibrationResult result;

    Eigen::Matrix3d& A = result.A;
    Eigen::Vector3d& b = result.b;
    double& magB = result.magB;
    double er;

    EllipsoidResult ellipsoid4 = correctEllipsoid4(x,y,z);
    A = ellipsoid4.Winv;
    b = ellipsoid4.V;
    magB = ellipsoid4.B;
    er = ellipsoid4.er;

    EllipsoidResult ellipsoid7 = correctEllipsoid7(x,y,z);
    if (ellipsoid7.er < er) {
        A = ellipsoid7.Winv;
        b = ellipsoid7.V;
        magB = ellipsoid7.B;
        er = ellipsoid7.er;
    }

    EllipsoidResult ellipsoid10 = correctEllipsoid10(x,y,z);
    if (ellipsoid10.er < er) {
        A = ellipsoid10.Winv;
        b = ellipsoid10.V;
        magB = ellipsoid10.B;
    }

    return result;
}

static CalibrationResult parameterizedfit(const std::string& str, 
                                        const Eigen::VectorXd& x, 
                                        const Eigen::VectorXd& y, 
                                        const Eigen::VectorXd& z)
{
    CalibrationResult result;
    EllipsoidResult eres;
    if (str == "eye") {
        eres = correctEllipsoid4(x, y, z);
    } else if (str == "diag") {
        eres = correctEllipsoid7(x, y, z);
    } else if (str == "sym") {
        eres = correctEllipsoid10(x, y, z);
    } else {  // auto
        result = bestfit(x, y, z);
    }
    if ((str == "eye") || (str == "diag") || (str == "sym")) {
        result.A = eres.Winv;
        result.b = eres.V;
        result.magB = eres.B;
    }

    return result;
}

static void validateAttributes(const MatrixXd& d)
{
    if (d.rows() == 0 || d.cols() != 3) {
        throw invalid_argument("Input must be a 2D matrix with 3 columns.");
    }
}

static void validateFitKind(const string& fitkind)
{
    const vector<string> validFitKinds = {"eye", "diag", "sym", "auto"};
    if (find(validFitKinds.begin(), validFitKinds.end(), fitkind) == validFitKinds.end()) {
        throw invalid_argument("Invalid fitkind value.");
    }
}

CalibrationResult magcal(const Eigen::MatrixXd d, const string& fitkind)
{

    CalibrationResult result;
    // Validate input
    validateAttributes(d);

    // 将输入数据转换为Eigen的VectorXd
    VectorXd x = d.col(0); // 第一列
    VectorXd y = d.col(1); // 第二列
    VectorXd z = d.col(2); // 第三列

    if (fitkind.empty()) {
        // 调用bestfit函数
        result = bestfit(x, y, z);
    } else {
        // 验证fitkind并调用parameterizedfit
        validateFitKind(fitkind);
        result = parameterizedfit(fitkind, x, y, z);
    }

    return result;
}

#ifdef MAG_TEST

int main(void)
{
    CalibrationResult result;
    // 正确的映射方式
    Eigen::VectorXd x = Eigen::Map<Eigen::VectorXd>(X_err.data(), X_err.size());  // 1行3列
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(Y_err.data(), Y_err.size());  // 1行3列
    Eigen::VectorXd z = Eigen::Map<Eigen::VectorXd>(Z_err.data(), Z_err.size());  // 1行3列


    // 将x, y, z转置并拼接到d矩阵
    Eigen::MatrixXd d(x.size(), 3);
    d << x, y, z;

/*
    // 打印结果
    // std::cout << "Matrix d:\n" << d << std::endl;
    

    Eigen::VectorXd v1 = smallestEigenVector(d);
    cout << "smallestEigenVector get :\n" << v1 << "\n" << endl;

    EllipsoidResult r4 = correctEllipsoid4(x, y, z);

    cout << "r4.Winv get :\n" << r4.Winv << "\n" << endl;
    cout << "r4.V get :\n" << r4.V << "\n" << endl;
    cout << "r4.B get :\n" << r4.B << "\n" << endl;
    cout << "r4.er get :\n" << r4.er << "\n" << endl;
    cout << "r4.ispd get :\n" << r4.ispd << "\n" << endl;


    Eigen::VectorXd v2 = residual(r4.Winv, r4.V, r4.B, d);
    cout << "v2 get :\n" << v2 << "\n" << endl;



    EllipsoidResult r7 = correctEllipsoid7(x, y, z);

    cout << "r7.Winv get :\n" << r7.Winv << "\n" << endl;
    cout << "r7.V get :\n" << r7.V << "\n" << endl;
    cout << "r7.B get :\n" << r7.B << "\n" << endl;
    cout << "r7.er get :\n" << r7.er << "\n" << endl;
    cout << "r7.ispd get :\n" << r7.ispd << "\n" << endl;

    EllipsoidResult r10 = correctEllipsoid10(x, y, z);

    cout << "r10.Winv get :\n" << r10.Winv << "\n" << endl;
    cout << "r10.V get :\n" << r10.V << "\n" << endl;
    cout << "r10.B get :\n" << r10.B << "\n" << endl;
    cout << "r10.er get :\n" << r10.er << "\n" << endl;
    cout << "r10.ispd get :\n" << r10.ispd << "\n" << endl;
*/

    result = magcal(d);
    cout << "result.A get :\n" << result.A << "\n" << endl;
    cout << "result.b get :\n" << result.b << "\n" << endl;
    cout << "result.magB get :\n" << result.magB << "\n" << endl;

    Eigen::MatrixXd dc = (d.rowwise() - result.b.transpose()) * result.A;
    cout << "dc get :\n" << dc << "\n" << endl;
    return 0;
}

#endif

