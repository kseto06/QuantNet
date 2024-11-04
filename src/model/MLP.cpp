#include "MLP.h"
#include "linalg.h"

#include <vector>
#include <map>
#include <__random/random_device.h>
#include <random>

namespace MLP {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::map<std::string, Matrix> matrixDict;

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
}