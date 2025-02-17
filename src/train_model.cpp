#include "model/linalg.h"
#include "model/HybridModel.h"
#include <vector>
#include <iostream>

/* TESTING STAGE */
int main() {
    // Generate sample data
    const int batch_size = 5;
    const int numUnits = 8;
    HybridModel::Tensor3D X_train = linalg::randn(10, 5, 3);  // 10 samples, 5 timesteps, 3 features
    const HybridModel::Matrix Y_train = linalg::randn(10, 1);  // 10 samples, 2 output classes

    //Init data and parameters for HybridModel
    HybridModel::init_data(X_train, Y_train, batch_size);
    HybridModel::init_hidden_units(numUnits);

    // Init model parameters
    const std::vector<std::string> layer_types = {"LSTM", "LSTM", "Relu", "Linear"}; //Neural network
    const std::vector<int> layer_dims = {static_cast<int>(X_train[0][0].size()), 12, 8, static_cast<int>(Y_train.size())}; //Neural network layers/features

    // Initialize the layers
    HybridModel::init_layers(layer_types, layer_dims);

    // Initialize the network parameters
    HybridModel::initialize_network();

    // Generate minibatches
    auto minibatches = HybridModel::generate_minibatches(X_train, Y_train, batch_size, 42);  // Batch size: 2, seed: 42

    // Model iteration through minibatches
    for (const auto& batch : minibatches) {
        auto& X_batch = std::get<0>(batch);
        auto& Y_batch = std::get<1>(batch); 

        // Forward prop
        HybridModel::forward_prop();
        std::cout << "Forward prop done" << std::endl;

        // Compute loss
        HybridModel::loss();
        std::cout << "Loss computed" << std::endl;

        // Backward prop
        HybridModel::back_prop();
        std::cout << "Backprop done.\n";
    }

    return 0;
}