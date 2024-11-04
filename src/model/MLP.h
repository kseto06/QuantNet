#ifndef MLP_H
#define MLP_H

#include <vector>
#include <map>

namespace MLP {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::map<std::string, Matrix> matrixDict;

    Matrix he_normalization(const int rows, const int cols);
    matrixDict init_mlp_params(const std::vector<int>& layer_dimensions, const int layer);
    std::tuple<Matrix, matrixDict> Dense(Matrix a_in, matrixDict& params, const std::function<Matrix(Matrix)>& activation, const int layer, matrixDict& cache);
};

#endif //MLP_H
