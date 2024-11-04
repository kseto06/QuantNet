#ifndef LSTMNETWORK_H
#define LSTMNETWORK_H

#include <vector>
#include <map>
#include <variant>

namespace LSTMNetwork {

    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::map<std::string, Matrix> matrixDict;

    //Forward prop
    typedef std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cacheTuple;

    //Variants for backprop
    typedef std::variant<Matrix, Tensor3D> variantTensor;
    typedef std::map<std::string, variantTensor> gradientDict;

    matrixDict init_params(const int n_input, const int n_hidden, const int n_output, const int layer);

    std::tuple<Tensor3D, Tensor3D, Tensor3D, std::tuple<std::vector<cacheTuple>, Tensor3D>>
    lstm_forward(const Tensor3D& x, const Matrix& a_initial, matrixDict& params, const int layer);

    gradientDict lstm_backprop(Tensor3D da, std::tuple<std::vector<cacheTuple>, Tensor3D> fwd_prop_cache, const int layer);
}

#endif //LSTMNETWORK_H
