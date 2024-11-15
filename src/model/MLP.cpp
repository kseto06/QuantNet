#include "MLP.h"
#include "linalg.h"

#include <vector>
#include <map>
#include <__random/random_device.h>
#include <random>

namespace MLP {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::map<std::string, Matrix> matrixDict; //Forward cache and gradients for mlp

    // Use W_y, a_next, b_y as Dense's weights, hidden state (a), biases
    Matrix he_normalization(const int rows, const int cols) {
        std::random_device rd;
        std::mt19937 gen(rd());

        double stdev = std::sqrt(2.0 / rows);
        std::normal_distribution<double> distrib(0, stdev);
        std::vector <std::vector <double> > result(rows, std::vector<double>(cols));
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result[i][j] = distrib(gen); //Random values for He
             }
        }
    }

    matrixDict init_mlp_params(const std::vector<int>& layer_dimensions, const int layer) {
        /*
        Inputs:
        layer_dimensions -- array containing the dimensions of each layer in the NN

        Outputs:
        params -- dictionary of params (W1, b1, W2, b2, ...)
        */
        //np.random.seed(3)
        std::map<std::string, Matrix> params;

        params["W"+std::to_string(layer)] = he_normalization(layer_dimensions[layer], layer_dimensions[layer-1]);
        // Bias matrix. Generates a matrix of shape[num units in current layer, 1 value]
        params["b"+std::to_string(layer)] = linalg::generateZeros(layer_dimensions[layer], 1);

        return params;
    }

    //Dense layer (MLP)
    std::tuple<Matrix, matrixDict> Dense(Matrix a_in, matrixDict& params, const std::function<Matrix(Matrix)>& activation, const int layer, matrixDict& cache) {
        const Matrix W = params["W"+std::to_string(layer)];
        Matrix b = params["b"+std::to_string(layer)];

        if (layer == 1) {
            a_in = linalg::transpose(a_in);
        }

        const Matrix Z = linalg::add(linalg::matmul(W, a_in), b);
        const Matrix a_out = activation(Z);

        cache["Z"+std::to_string(layer)] = Z;
        cache["A"+std::to_string(layer)] = a_out;
        cache["W"+std::to_string(layer)] = W;
        cache["b"+std::to_string(layer)] = b;

        return std::make_tuple(a_out, cache);
    }

    //Backprop one step (MLP)
    matrixDict mlp_backward(Matrix a_in, Matrix dA, Matrix targets, matrixDict mlp_cache, const int layer, const std::function<Matrix(Matrix)>& prime_activation) {
        //Z derivative
        const Matrix dZ = linalg::elementMultiply(dA, prime_activation(mlp_cache["Z"+std::to_string(layer)]));

        //(W)eight derivative
        const Matrix dW = (layer > 1) ? linalg::matmul(dZ, mlp_cache["A"+std::to_string(layer)]) : linalg::matmul(dZ, a_in); //Use the original input for the last layer

        // Update B and A gradients
        const Matrix dB = linalg::sum(dZ, 1); //Sum over dZ's columns
        const Matrix dA_prev = linalg::matmul(mlp_cache["W"+std::to_string(layer)], dZ);

        // Storing gradients to return:
        matrixDict gradients;
        gradients["dZ"+std::to_string(layer)] = dZ;
        gradients["dW"+std::to_string(layer)] = dW;
        gradients["db"+std::to_string(layer)] = dB;
        gradients["dA"+std::to_string(layer)] = dA_prev;

        return gradients;
    }


}