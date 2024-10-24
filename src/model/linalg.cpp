#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <__random/random_device.h>

class linalg {
    typedef std::vector<std::vector<double>> Matrix;

public:
    std::string shape(const Matrix &m) {
        return std::to_string(m.size()) + ", " +std::to_string(m[0].size());
    }

    Matrix generateZeros(const int a, const int b) {
        Matrix result(a, std::vector<double>(b));

        for (size_t i = 0; i < a; i++) {
            for (size_t j = 0; j < b; j++) {
                result[i][j] = 0.0;
            }
        }

        return result;
    }

    //This function computes the dot product of two vectors, outputs scalar
    double dot(const std::vector <double> &a, const std::vector <double> &b) {
        /*
        a[0] * b[0] + a[1] * b[1] +...+ a[i] * b[i]
        */

        if (a.size() != b.size()) {
            throw std::invalid_argument("Vector shape mismatch in dot product");
        }
        
        double dotProduct = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            dotProduct += a[i] * b[i];
        }
        return dotProduct;
    }
    //This function computes the matmul product of two matrices
    Matrix &matmul(const Matrix &a, const Matrix &b) {
        /*
        Result[i][j]=
            v=0
            ∑     (MatrixA[i][v]×MatrixB[v][j])
            n−1
        */
        //Ensure same shape
        if (a[0].size() != b.size()|| a.size() != b[0].size()) {
            throw std::invalid_argument("Matrices have different shapes. a_shape: " + shape(a) + " b shape: " + shape(b));
        }
        // Generate array of zeros
        Matrix product = generateZeros(a.size(), b[0].size());

        for (size_t i = 0; i < a.size(); i++) {
            // Matrix multiplication:
            for (size_t j = 0; j < b[0].size(); j++) {
                double sum = 0; //sum of each row
                for (size_t v = 0; v < a[i].size(); v++) {
                    sum += a[i][v] * b[v][j];
                }
                product[i][j] = sum; // Append the product
            }
        }
        return product;
    }

    // Element wise addition
    Matrix add(const Matrix &a, const Matrix &b) {
        if (a.size() != b.size() || a[0].size() != b[0].size()) {
            throw std::invalid_argument("Matrices not the same shape for addition");
        }
        // Generate array of zeros
        Matrix result = generateZeros(a.size(), b[0].size());

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    // Element wise subtraction
    Matrix  subtract(const Matrix &a, const Matrix &b) {
        if (a.size() != b.size() || a[0].size() != b[0].size()) {
            throw std::invalid_argument("Matrices not the same shape for addition");
        }
        // Generate array of zeros
        Matrix result = generateZeros(a.size(), b[0].size());

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    Matrix transpose(const Matrix &m) {
        // Initialize transposed matrix with swapped dimensions
        Matrix transposed(m[0].size(), std::vector<double>(m.size()));

        for (size_t i = 0; i < m.size(); i++) {
            for (size_t j = 0; j < m[0].size(); j++) {
                transposed[j][i] = m[i][j];
            }
        }
        return transposed;
    }

    Matrix pow(const Matrix &m, const double exponent) {
        // Copy the current matrix
        Matrix result(m.size(), std::vector<double>(m[0].size()));

        // Element-wise power
        for (size_t i = 0; i < m.size(); i++) {
            for (size_t j = 0; j < m[0].size(); j++) {
                result[i][j] = std::pow(m[i][j], exponent);
            }
        }
        return result;
    }

    Matrix sqrt(const Matrix &m) {
        return pow(m, 0.5);
    }

    Matrix sum(const Matrix &m, const int axis) {
        //This function assumes keepdims = True
        if (axis == 0) {
            // Sum along columns, index 1 represents sum.
            Matrix colSum(1, std::vector<double>(m[0].size(), 0.0));

            for (size_t j = 0; j < m[0].size(); j++) {
                for (size_t i = 0; i < m.size(); i++) {
                    colSum[0][j] += m[i][j];
                }
            }
            return colSum;
        }
        else {
            //Sum along rows
            Matrix rowSum(m.size(), std::vector<double>(1, 0.0));

            for (size_t i = 0; i < m.size(); i++) {
                for (size_t j = 0; j < m[0].size(); j++) {
                    rowSum[i][0] += m[i][j];
                }
            }
            return rowSum;
        }
    }

    //Scalar multiplication
    Matrix scalarMultiply(const double scalar, const Matrix &m) {
        Matrix result = m;
        for (size_t i = 0; i < m.size(); i++) {
            for (size_t j = 0; j < m[0].size(); j++) {
                result[i][j] = scalar * m[i][j];
            }
        }
        return result;
    }

    //Element-wise multiplication
    Matrix elementMultiply(const Matrix &a, const Matrix &b) {
        Matrix result(a.size(), std::vector<double>(a[0].size(), 0.0));

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                result[i][j] == a[i][j] * b[i][j];
            }
        }
        return result;
    }

    //Element-wise division
    Matrix division(const Matrix &a, const Matrix &b) {
        // Ensure dimensions match -- broadcasting b in L2 Norm
        if (a[0].size() != b[0].size()) {
            throw std::invalid_argument("Shape mismatch in element-wise division");
        }

        Matrix result(a.size(), std::vector<double>(a[0].size(), 0.0));

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                if (b[i][j] == 0) {
                    result[i][j] = 0; //Prevents division by zero
                }
                else {
                    result[i][j] = a[i][j] / b[i][j];
                }
            }
        }
        return result;
    }

    double randn() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::normal_distribution<double> distrib(0, 1);
        return distrib(gen);
    }

    // randn to generate a vector of random numbers
    std::vector<double> randn(const int n) {
        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = randn();
        }
        return result;
    }

    // randn to generate a matrix of random numbers
    Matrix randn(const int rows, const int cols) {
        Matrix result(rows, std::vector<double>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = randn();
            }
        }
        return result;
    }

};
