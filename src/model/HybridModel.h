#ifndef HYBRIDMODEL_H
#define HYBRIDMODEL_H

#include <vector>

namespace HybridModel {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::tuple<Tensor3D, Matrix> minibatch;

    std::vector<minibatch> generate_minibatches(const Tensor3D& X, const Matrix& Y, int batch_size, int seed);
    double MSE(const std::vector<double>& pred, const std::vector<double>& target);
    void init_data(const std::variant<Matrix, Tensor3D>& X, const Matrix& Y, const int batch_size);
    void init_layers(const std::vector<std::string>& layer_type, const std::vector<int>& layer_dim);
    void init_hidden_units(const int numUnits);
    void initialize_network();
    Matrix reshape_last_timestep(const Tensor3D& hidden_state);
    void forward_prop(std::variant<Tensor3D, Matrix> x_train); //x_train = x_batch
    void loss(Matrix y_train); //y_train = y_batch
    void back_prop();
}

#endif //HYBRIDMODEL_H
