#include <vector>

namespace activations {
    typedef std::vector<std::vector<double>> Matrix;

        //Apply linear activation to a matrix
        Matrix linear(const Matrix &m) {
            return m;
        }

        //Linear prime to a matrix
        Matrix linear_prime(const Matrix &m) {
            std::vector <std::vector <double> > result = m;

            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = 1.0;
                }
            }

            return result;
        }

        //Apply ReLU activation function to a matrix
        Matrix relu(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = std::max(0.0, result[i][j]);
                }
            }
            return result;
        }

        Matrix relu_prime(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    if (result[i][j] > 0.0) {
                        result[i][j] = 1.0;
                    } else {
                        result[i][j] = 0.0;
                    }
                }
            }
            return result;
        }


        Matrix sigmoid(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = 1 / (1+std::exp(-m[i][j]));
                }
            }
            return result;
        }

        Matrix sigmoid_prime(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = m[i][j] * (1 - m[i][j]);
                }
            }
            return result;
        }

        Matrix tanh(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = std::tanh(m[i][j]);
                }
            }
            return result;
        }

        Matrix tanh_prime(const Matrix &m) {
            Matrix result = m;
            for (size_t i = 0; i < result.size(); i++) {
                for (size_t j = 0; j < result[0].size(); j++) {
                    result[i][j] = 1 - std::pow(std::tanh(m[i][j]), 2);
                }
            }
        }
};