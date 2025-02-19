#ifndef LINALG_H
#define LINALG_H

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>

namespace linalg {
    // Type definitions
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;

    // Function declarations
    std::string shape(const Matrix &m);
    std::string shapeTensor(const Tensor3D &m);

    std::vector<double> generateZeros(const int n);
    Matrix generateZeros(const int rows, const int cols);
    Tensor3D generateZeros(const int rows, const int timesteps, const int cols);

    std::vector<double> generateOnes(const int n);
    Matrix generateOnes(const int rows, const int cols);
    Tensor3D generateOnes(const int rows, const int timesteps, const int cols);

    double dot(const std::vector<double> &a, const std::vector<double> &b);
    Matrix matmul(const Matrix &a, const Matrix &b);
    Matrix add(const Matrix &a, const Matrix &b);
    Matrix add(const Matrix &a, const double s);
    Matrix subtract(const Matrix &a, const Matrix &b);
    Matrix transpose(const Matrix &m);
    Matrix pow(const Matrix &m, const double exponent);
    Matrix sqrt(const Matrix &m);
    Matrix sum(const Matrix &m, const int axis);
    Matrix scalarMultiply(const double scalar, const Matrix &m);
    Matrix elementMultiply(const Matrix &a, const Matrix &b);
    Matrix division(const Matrix &a, const Matrix &b);
    Matrix division(const Matrix& a, const int s);

    double randnum();
    std::vector<double> randn(const int n);
    Matrix randn(const int rows, const int cols);
    Tensor3D randn(const int rows, const int timesteps, const int cols);
    Matrix sliceCols(const Matrix& mat, size_t start_col, size_t end_col);
    std::vector<double> reshape(const Matrix& m);
    Matrix reshape(const std::vector<double> v);

    void printVector(const std::vector<double>& vec);
    void printMatrix(const Matrix& mat);
    void printTensor3D(const Tensor3D& ten);

}

#endif // LINALG_H
