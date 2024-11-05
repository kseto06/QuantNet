#ifndef HYBRIDMODEL_H
#define HYBRIDMODEL_H

#include <vector>

namespace HybridModel {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    
    double MSE(const std::vector<double>& pred, const std::vector<double>& target);
    void init_layers(const std::vector<std::string>& layer_type, const std::vector<int>& layer_dim);
    void initialize_network();
    Matrix reshape_last_timestep(const Tensor3D& hidden_state);
    void forward_prop();
    void loss();
    void back_prop();
}



#endif //HYBRIDMODEL_H
