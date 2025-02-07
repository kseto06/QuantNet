#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <__random/random_device.h>
#include "linalg.h"

namespace linalg {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;

    std::string shape(const Matrix &m) {
        return std::to_string(m.size()) + ", " +std::to_string(m[0].size());
    }

    std::string shapeTensor(const Tensor3D &m) {
        return std::to_string(m.size()) + ", " + std::to_string(m[0].size()) + ", " + std::to_string(m[0][0].size());
    }

    //Vector generate zeros:
    std::vector<double> generateZeros(const int n) {
        std::vector<double> zero_vector(n, 0.0);
        return zero_vector;
    }

    //@Overload: Matrix generate zeros
    Matrix generateZeros(const int rows, const int cols) {
        Matrix result(rows, std::vector<double>(cols, 0.0));
        return result;
    }

    //@Overload: Tensor3D generate zeros
    Tensor3D generateZeros(const int rows, const int timesteps, const int cols) {
        Tensor3D result(rows, Matrix(timesteps, std::vector<double>(cols, 0.0)));
        return result;
    }

    //Vector generate ones:
    std::vector<double> generateOnes(const int n) {
        std::vector<double> one_vector(n, 1.0);
        return one_vector;
    }

    //@Overload: Matrix generate ones
    Matrix generateOnes(const int rows, const int cols) {
        Matrix result(rows, std::vector<double>(cols, 1.0));
        return result;
    }

    //@Overload: Tensor3D generate ones
    Tensor3D generateOnes(const int rows, const int timesteps, const int cols) {
        Tensor3D result(rows, Matrix(timesteps, std::vector<double>(cols, 1.0)));
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
    Matrix matmul(const Matrix &a, const Matrix &b) {
        /*
        Result[i][j]=
            v=0
            ∑     (MatrixA[i][v]×MatrixB[v][j])
            n−1
        */
        //Ensure same shape
        if (a[0].size() != b.size()) {
            //throw std::invalid_argument("Matrices have different shapes for matmul. a_shape: " + shape(a) + " b shape: " + shape(b));
        }
        // Generate array of zeros
        Matrix product = generateZeros(a.size(), b[0].size());

        // Matrix multiplication
        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < b[0].size(); j++) {
                double sum = 0; // Sum of each row
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
        if (a.size() != b.size()) {
            //throw std::invalid_argument("Matrices not the same shape for addition. a_shape: " + shape(a) + " b shape: " + shape(b));
        }

        // Generate array of zeros
        Matrix result = generateZeros(a.size(), a[0].size());

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                // Add broadcasting for weights and biases
                if (b[0].size() == 1) {
                    result[i][j] = a[i][j] + b[i][0];
                } else {
                    result[i][j] = a[i][j] + b[i][j];
                }
            }
        }
        return result;
    }

    // Element wise subtraction
    Matrix subtract(const Matrix &a, const Matrix &b) {
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
        if (a.size() != b.size() || (a.size() > 0 && b.size() > 0 && a[0].size() != b[0].size())) {
            std::string error_message = "Error in linalg::elementMultiply: Dimension mismatch.\n";
            error_message += "Shape of matrix 'a': " + std::to_string(a.size()) + "x" + (a.empty() ? "0" : std::to_string(a[0].size())) + "\n";
            error_message += "Shape of matrix 'b': " + std::to_string(b.size()) + "x" + (b.empty() ? "0" : std::to_string(b[0].size()));
            //throw std::invalid_argument(error_message); // Throw exception if dimensions don't match
        }

        Matrix result(a.size(), std::vector<double>(a[0].size(), 0.0));

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                result[i][j] = a[i][j] * b[i][j];
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

    //Element-wise division of a matrix by a scalar
    Matrix division(const Matrix &a, const int s) {
        Matrix result(a.size(), std::vector<double>(a[0].size(), 0.0));

        //Prevent division by zero
        if (s == 0) {
            return a;
        }

        for (size_t i = 0; i < a.size(); i++) {
            for (size_t j = 0; j < a[0].size(); j++) {
                result[i][j] = a[i][j] / s;
            }
        }
        return result;
    }

    double randnum() {
        constexpr int SEED = 0; //Seed can be changed for reproducibility
        static std::random_device rd;
        static std::mt19937 gen(SEED);
        static std::normal_distribution<double> distrib(0, 1);
        return distrib(gen);
    }

    // randn to generate a vector of random numbers
    std::vector<double> randn(const int n) {
        std::vector<double> result(n);
        for (int i = 0; i < n; i++) {
            result[i] = randnum();
        }
        return result;
    }

    // randn to generate a matrix of random numbers
    Matrix randn(const int rows, const int cols) {
        Matrix result(rows, std::vector<double>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[i][j] = randnum();
            }
        }
        return result;
    }

    Matrix sliceCols(const Matrix& mat, size_t start_col, size_t end_col) {
        size_t rows = mat.size();
        size_t cols = mat[0].size();

        // Ensure end_col is within bounds, start_col < end_col
        if (end_col > cols || start_col >= end_col) {
            throw std::invalid_argument("Invalid column range for slicing.");
        }

        Matrix sliced(rows, std::vector<double>(end_col - start_col));

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = start_col; j < end_col; j++) {
                sliced[i][j - start_col] = mat[i][j];
            }
        }

        return sliced;
    }

    //Reshape a (m, 1) Matrix --> (m) vector
    std::vector<double> reshape(const Matrix& m) {
        std::vector<double> vector(m.size());
        for (size_t i = 0; i < m.size(); i++) {
            vector.push_back(m[i][0]);
        }
        return vector;
    }

    //Reshape a (m) vector --> (m, 1) Matrix
    Matrix reshape(const std::vector<double>& v) {
        const int m = v.size();
        Matrix matrix(m, std::vector<double>(1));

        for (size_t i = 0; i < m; i++) {
            matrix[i][0] = v[i];
        }

        return matrix;
    }

    // Function to print a vector
    void printVector(const std::vector<double>& vec) {
        std::cout << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i < vec.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    // Function to print a matrix:
    void printMatrix(const Matrix& mat) {
        std::cout << "[";
        for (int i = 0; i < mat.size(); i++) {
            for (int j = 0; j < mat[0].size(); j++) {
                std::cout << mat[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    // Function to print a Tensor3D:
    void printTensor3D(const Tensor3D& ten) {
        std::cout << "[";
        for (int i = 0; i < ten.size(); i++) {
            for (int j = 0; j < ten[0].size(); j++) {
                for (int k = 0; k < ten[0][0].size(); k++) {
                    std::cout << ten[i][j][k] << " ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
};
