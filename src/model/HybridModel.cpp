#include "HybridModel.h"
#include "MLP.h"
#include "LSTMNetwork.h"
#include "activations.h"

#include <cmath>
#include <vector>
#include <map>
#include <random>

#include "linalg.h"

namespace HybridModel {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::variant<Matrix, Tensor3D> variantTensor;
    typedef std::map<std::string, Matrix> matrixDict; //Global params
    typedef std::vector<matrixDict> MLPCache; //cache for forward prop
    typedef std::tuple<Tensor3D, Matrix> minibatch;

    //LSTM
    typedef std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cacheTuple;
    typedef std::tuple<Tensor3D, Tensor3D, Tensor3D, std::tuple<std::vector<cacheTuple>, Tensor3D>> LSTMCache;

    //Backprop
    //Variant since it can be either a Tensor3D gradient with timesteps or Matrix gradients
    typedef std::map<std::string, variantTensor> gradientDict;

    //Unified cache structure
    struct UnifiedCache {
        std::vector<std::variant<LSTMCache, matrixDict>> cache;
    };

    //Unified gradient structure
    struct UnifiedGradients {
        std::vector<std::variant<gradientDict, matrixDict>> grads;
    };

    //Anonymous namespace for private variables
    namespace {
        //Default layer types (i.e. Dense LSTM) and layer dimensions
        std::vector<std::string> layer_types = {};
        std::vector<int> layer_dims = {};
        std::vector<matrixDict> layer_params;

        //Forward prop variables
        UnifiedCache cache;
        Matrix finalPrediction; //Linear output matrix, shape(m,1)

        //Loss
        double accumulated_loss = 0.0;

        //Data, x_train and y_train. NOTE: x_train and y_train have to be generated by minibatches
        variantTensor x_train;
        Matrix y_train = {{}}; //shape (m,1)
        constexpr int BATCH_SIZE = 64;

        //Backprop variables
        UnifiedGradients grads;
    }

    // Minibatch generation
    std::vector<minibatch> generate_minibatches(Tensor3D X, Matrix Y, int batch_size, int seed) {
        size_t m = X.size();

        //Generate a permutation:
        std::vector<int> permutation(m);
        for (int i = 0; i < m; i++) {
            permutation[i] = i;
        }
        std::mt19937 perm(seed);
        std::shuffle(permutation.begin(), permutation.end(), perm);

        //Shuffle the data permutations:
        Tensor3D shuffled_X(m, Matrix(X[0].size(), std::vector<double>(X[0][0].size())));
        Matrix shuffled_Y(m, std::vector<double>(1));
        for (int i = 0; i < m; i++) {
            int perm_index = permutation[i];
            //Ensure bounds
            if (perm_index >= m || perm_index < 0) {
                throw std::out_of_range("Index out of range");
            }

            shuffled_X[i] = X[perm_index];
            shuffled_Y[i] = Y[perm_index];
        }

        size_t num_minibatches = floor(m / batch_size);
        std::vector<minibatch> minibatches;
        for (size_t k = 0; k < num_minibatches; k += batch_size) {
            //Prevent overshooting the count on the last batch
            int end = std::min(k + batch_size, num_minibatches);

            Tensor3D minibatch_X(batch_size, Matrix(X[0].size(), std::vector<double>(X[0][0].size())));
            Matrix minibatch_Y(m, std::vector<double>(1));
            //Create batches
            for (int i = k; i < end; i++) {
                X[i - k] = X[i];
                Y[i - k] = Y[i];
            }

            minibatches.emplace_back(std::move(minibatch_X), std::move(minibatch_Y));
        }
        return minibatches;
    }

    // MSE loss function
    double MSE(const std::vector<double>& pred, const std::vector<double>& target) {
        double loss = 0.0;
        for (size_t i = 0; i < pred.size(); i++) {
            loss += std::pow(pred[i] - target[i], 2);
        }
        return loss/(2*pred.size());
    }

    //Layer types and dimensions (setters)
    void init_layers(const std::vector<std::string>& layer_type, const std::vector<int>& layer_dim) {
        layer_types = layer_type;
        layer_dims = layer_dim;
    }

    //LSTM Network initialization
    void initialize_network() {
        //NOTE: layer_type and layer_dims should have the same shape
        for (int i = 1; i <= layer_types.size(); i++) {
            matrixDict current_params;
            if (layer_types[i] == "LSTM") {
                current_params = LSTMNetwork::init_params(BATCH_SIZE, std::get<Tensor3D>(x_train)[0].size(), layer_dims[i], i); //Output matches the input shape?
            } else if (layer_types[i] == "Relu" || layer_types[i] == "Linear") {
                current_params = MLP::init_mlp_params(layer_dims, i);
            }
            layer_params.push_back(current_params);
        }
    }

    //Tensor3D --> Matrix conversion based on last timestep output
    Matrix reshape_last_timestep(const Tensor3D& hidden_state) {
        int batch_size = hidden_state.size();
        int hidden_units = hidden_state[0][0].size();
        Matrix reshaped_matrix(batch_size, std::vector<double>(hidden_units));

        // Extract the last timestep for each example in the batch
        for (int i = 0; i < batch_size; ++i) {
            if (hidden_state[i].empty()) {
                throw std::invalid_argument("Hidden state is empty");
            }

            reshaped_matrix[i] = hidden_state[i].back();  // return the last timestep in the sequence
        }
        return reshaped_matrix;
    }

    //Matrix --> Tensor3D conversion with number of timesteps initialized in x_train
    Tensor3D reshape_last_timestep(const Matrix& hidden_state) {
        int batch_size = hidden_state.size();
        int hidden_units = hidden_state[0].size();
        const int TIMESTEPS = std::get<Tensor3D>(x_train)[0].size();
        Tensor3D reshaped_tensor(batch_size, Matrix(TIMESTEPS, std::vector<double>(hidden_units, 0.0)));

        // Reshape:
        for (int i = 0; i < batch_size; i++) {
            for (int t = 0; t < TIMESTEPS; t++) {
                reshaped_tensor[i][t] = hidden_state[i];
            }
        }

        return reshaped_tensor;
    }

    void forward_prop() {
        /*
        NOTE: Right now, function assumes that the first inputs are LSTMs and last inputs are MLP.
              - e.g: Relu->Relu->LSTM->LSTM is not supported, because LSTMs are placed last
              - e.g: LSTM->LSTM->Relu->Linear is supported, because LSTMs are placed before MLP in the network
              Architectures that are "mixed" is not supported
              - e.g: LSTM->Relu->LSTM->Linear
         */
        Matrix Wy = layer_params[0]["Wy1"];
        int n_a = Wy[0].size();

        //MLP
        Matrix a_out;

        //LSTM
        Matrix a_initial = linalg::generateZeros(std::get<Tensor3D>(x_train).size(), n_a); //Initially, a0 is a Matrix of zeros with shape (m, n_a)
        Tensor3D new_x_state;
        Tensor3D new_hidden_state;

        for (int i = 1; i <= layer_types.size(); i++) {
            if (layer_types[i] == "LSTM") {
                if (i == 1) {
                    //Initialize parameters in the function and forward prop through the network once
                    LSTMCache current_lstm_tuple = LSTMNetwork::lstm_forward(std::get<Tensor3D>(x_train), a_initial, layer_params[i], i);
                    new_x_state = std::get<1>(std::get<3>(current_lstm_tuple));
                    new_hidden_state = std::get<0>(current_lstm_tuple);
                    cache.cache.push_back(current_lstm_tuple);
                } else {
                    LSTMCache current_lstm_tuple = LSTMNetwork::lstm_forward(new_x_state, reshape_last_timestep(new_hidden_state), layer_params[i], i);
                    new_x_state = std::get<1>(std::get<3>(current_lstm_tuple));
                    new_hidden_state = std::get<0>(current_lstm_tuple);
                    cache.cache.push_back(current_lstm_tuple);
                }
            } else if (layer_types[i] == "Relu") {
                // Reshape a_out using the last timestepped hidden state from LSTM_forward
                if (layer_types[i-1] == "LSTM" && i != 1) {
                    a_out = reshape_last_timestep(new_hidden_state);
                }

                if (i == 1) {
                    //Input x is a Matrix
                    std::tuple<Matrix, matrixDict> current_dense_tuple = MLP::Dense(std::get<Matrix>(x_train), layer_params[i], activations::relu, i, std::get<matrixDict>(cache.cache[i]));
                    a_out = std::get<0>(current_dense_tuple);
                    matrixDict current_mlp_cache = std::get<1>(current_dense_tuple);
                    cache.cache.push_back(current_mlp_cache);
                } else {
                    std::tuple<Matrix, matrixDict> current_dense_tuple = MLP::Dense(a_out, layer_params[i], activations::relu, i, std::get<matrixDict>(cache.cache[i]));
                    a_out = std::get<0>(current_dense_tuple);
                    matrixDict current_mlp_cache = std::get<1>(current_dense_tuple);
                    cache.cache.push_back(current_mlp_cache);
                }
            } else if (layer_types[i] == "Linear") {
                // Reshape a_out using the last timestepped hidden state from LSTM_forward
                if (layer_types[i-1] == "LSTM" && i != 1) {
                    a_out = reshape_last_timestep(new_hidden_state);
                }

                std::tuple<Matrix, matrixDict> current_dense_tuple = MLP::Dense(a_out, layer_params[i], activations::linear, i, std::get<matrixDict>(cache.cache[i]));
                a_out = std::get<0>(current_dense_tuple);
                matrixDict current_mlp_cache = std::get<1>(current_dense_tuple);
                cache.cache.push_back(current_mlp_cache);
            }
        }
        //Set the final prediction matrix
        finalPrediction = a_out;
    }

    void loss() {
        std::vector<double> predictions = linalg::reshape(finalPrediction);
        std::vector<double> targets = linalg::reshape(y_train);

        //predictions and current y_train are of the same mini-batch (BATCH_SIZE = 64):
        accumulated_loss += MSE(predictions, targets);
    }

    void back_prop() {
        gradientDict gradients;
        constexpr int L = layer_types.size(); //num of layers
        constexpr int m = std::get<Tensor3D>(x_train).size();
        Matrix a_in_matrix = reshape_last_timestep(std::get<Tensor3D>(x_train));

        // Derivatives
        Matrix dA_matrix;
        if (std::holds_alternative<matrixDict>(cache.cache[L])) {
            //Access the cache at L
            matrixDict& layer_cache = std::get<matrixDict>(cache.cache[L]);

            // Check if the key exists
            auto item = layer_cache.find("A"+std::to_string(L));
            if (item != layer_cache.end()) {
                dA_matrix = item -> second;
            }

            dA_matrix = linalg::division(linalg::subtract(dA_matrix, y_train), m); //Init gradient for the last layer (derivative of loss function)
        }
        Tensor3D dA_tensor; //To store reshaped LSTM gradients

        for (int layer = L; layer >= 1; layer--) {
            if (layer_types[layer] == "LSTM") {
                if (layer == L) {
                    continue; //Skip, assume last layer is always a linear/MLP output
                }

                //Reshape from Matrix to Tensor3D if last backpropagated layer wasn't LSTM with Tensor3D
                if (layer_types[layer-1] == "Relu" || layer_types[layer-1] == "Linear") {
                    dA_tensor = reshape_last_timestep(dA_matrix);
                }

                if (std::holds_alternative<LSTMCache>(cache.cache[layer])) { //Check for correct type
                    //Get the current LSTM cache
                    LSTMCache lstm_cache = std::get<LSTMCache>(cache.cache[layer]);
                    gradientDict current_lstm_grads = LSTMNetwork::lstm_backprop(dA_tensor, std::tuple<std::vector<LSTMNetwork::cacheTuple>, LSTMNetwork::Tensor3D>(
                        std::make_tuple<lstm_cache, std::get<Tensor3D>(x_train)>), layer);

                    // Update the new activation derivative
                    dA_tensor = std::get<Tensor3D>(current_lstm_grads["da0"+std::to_string(layer)]);

                    //Store gradients
                    grads.grads.push_back(current_lstm_grads);
                }

            } else if (layer_types[layer] == "Relu" || layer_types[layer] == "Linear") {
                if (layer == L) {
                    continue;
                }
                // Reshape dA to a matrix using the last timestepped hidden state from LSTM gradients
                if (layer_types[layer-1] == "LSTM") {
                    dA_matrix = reshape_last_timestep(dA_tensor);
                }

                //Compute gradients
                matrixDict current_mlp_grads = MLP::mlp_backward(
                    a_in_matrix, dA_matrix, y_train,
                    std::get<matrixDict>(cache.cache[layer]), layer,
                    (layer_types[layer] == "Relu") ? activations::relu : activations::linear); //Ternary operator between Relu and Linear

                //Store gradients
                grads.grads.push_back(current_mlp_grads);
            }
        }

    }
}
