#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

namespace activations {
    //Type definitions
    typedef std::vector<std::vector<double>> Matrix;

    //Function declarations
    Matrix linear(const Matrix &m);
    Matrix linear_prime(const Matrix &m);

    Matrix relu(const Matrix &m);
    Matrix relu_prime(const Matrix &m);

    Matrix sigmoid(const Matrix &m);
    Matrix sigmoid_prime(const Matrix &m);

    Matrix tanh(const Matrix &m);
    Matrix tanh_prime(const Matrix &m);
};

#endif //ACTIVATIONS_H
