#include "model/linalg.h"
#include "model/HybridModel.h"
#include <vector>
#include <iostream>

/* TESTING STAGE */
int main() {
    // Init model parameters
    std::vector<std::string> layer_types = {"LSTM", "LSTM", "Relu", "Linear"};
    std::vector<int> layer_dims = {3, 5, 2, 1};

    // Initialize the layers
    HybridModel::init_layers(layer_types, layer_dims);

    // Generate data
    HybridModel::Tensor3D X_train(10, std::vector<std::vector<double>>(5, std::vector<double>(3, 0.5)));  // 10 samples, 5 timesteps, 3 features
    HybridModel::Matrix Y_train(10, std::vector<double>(2, 1.0));  // 10 samples, 2 output classes
    HybridModel::init_data(X_train, Y_train);//Input into HybridModel class

    // Initialize the network parameters
    HybridModel::initialize_network();

    // Generate minibatches
    auto minibatches = HybridModel::generate_minibatches(X_train, Y_train, 2, 42);  // Batch size: 2, seed: 42

    // Model iteration through minibatches
    for (const auto& batch : minibatches) {
        auto& X_batch = std::get<0>(batch);  // Input tensor
        auto& Y_batch = std::get<1>(batch);  // Output matrix

        // Reshape the last timestep for testing purposes
        // HybridModel::Matrix reshaped_X = HybridModel::reshape_last_timestep(X_batch);
        // std::cout << "Reshaped X: " << reshaped_X.size() << " x " << reshaped_X[0].size() << std::endl;


        // Forward prop
        // HybridModel::forward_prop();
        // std::cout << "Forward propagation completed.\n";
        //
        // // Compute loss
        // HybridModel::loss();
        // std::cout << "Loss computation completed.\n";
        //
        // // Backward prop
        // HybridModel::back_prop();
        // std::cout << "Backward propagation completed.\n";
    }

    // Final MSE loss
    // std::vector<double> pred = {0.9, 0.8};
    // std::vector<double> target = {1.0, 1.0};
    // double mse = HybridModel::MSE(pred, target);
    // std::cout << "MSE: " << mse << std::endl;

    return 0;
}

