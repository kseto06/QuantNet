#include <vector>
#include <map>
#include "LSTMCell.h"
#include "linalg.h"

namespace LSTMNetwork {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::map<std::string, Matrix> matrixDict;

    //Forward prop
    typedef std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cacheTuple;
    
    //Variants for backprop
    typedef std::variant<Matrix, Tensor3D> variantTensor;
    typedef std::map<std::string, variantTensor> gradientDict;

    matrixDict init_params(const int n_input, const int n_hidden, const int n_output) {
            //NOTE: n represents the columns / num of features in the data
            matrixDict params;

            //Initialize parameters to have small values
            //NOTE: We might need to transpose all these values
            //Forget gate:
            params["Wf"] = linalg::scalarMultiply(0.01, linalg::randn(n_hidden, n_hidden+n_input));
            params["bf"] = linalg::generateZeros(n_input, 1);

            //Update (input) gate:
            params["Wi"] = linalg::scalarMultiply(0.01, linalg::randn(n_hidden, n_hidden+n_input));
            params["bi"] = linalg::generateZeros(n_input, 1);

            //Candidate/memory cells
            params["Wc"] = linalg::scalarMultiply(0.01, linalg::randn(n_hidden, n_hidden+n_input));
            params["bc"] = linalg::generateZeros(n_input, 1);

            //Output gate:
            params["Wo"] = linalg::scalarMultiply(0.01, linalg::randn(n_hidden, n_hidden+n_input));
            params["bo"] = linalg::generateZeros(n_input, 1);

            //Predictions
            params["Wy"] = linalg::scalarMultiply(0.01, linalg::randn(n_output, n_hidden));
            params["by"] = linalg::generateZeros(n_output, 1);

            return params;
        }

    //Iterate through each cell at their respective timesteps
    std::tuple<Tensor3D, Tensor3D, Tensor3D, std::tuple<std::vector<cacheTuple>, Tensor3D>>
    lstm_forward(const Tensor3D& x, const Matrix& a_initial, matrixDict& params) {
            /* Inputs:
             * - x: input data, 3D Tensor of shape (num exs, num feats, timestep (days))
             * - a_initial: Initial hidden state
             * - parameters: map of weights and biases
             */

            //NOTE: if hybrid with MLP, cache may not be empty.
            std::vector <cacheTuple> cache;

            Matrix Wy = params["Wy"]; //Get the weight matrix for the prediction

            //Init shapes. NOTE: n_a, n_y might need to be reversed
            const int m = x.size(), n_x = x[0][0].size(), timesteps = x[0].size(), n_y = Wy.size(), n_a = Wy[0].size();

            // Init states
            Tensor3D hidden_state = linalg::generateZeros(m, timesteps, n_a);
            Tensor3D candidate = linalg::generateZeros(m, timesteps, n_a);
            Tensor3D prediction = linalg::generateZeros(m, timesteps, n_y);

            // Init matrices for hidden states at timesteps
            Matrix a_next = a_initial;
            Matrix c_next = linalg::generateZeros(m, n_a);

            //Forward pass for every timestep
            for (size_t timestep = 0; timestep < timesteps; timestep++) {
                // Slice the input data at the specific timestep:
                Matrix x_t(m, std::vector<double>(n_x));
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        x_t[i][j] = x[i][timestep][j];
                    }
                }

                //Compute the matrices and parameters for the current timestep cell
                std::tuple< Matrix, Matrix, Matrix, cacheTuple >
                cell_state = LSTMCell::lstm_cell_forward(x_t, a_next, c_next, params);

                //Extract the values of the current timestep cell
                a_next = std::get<0>(cell_state), c_next = std::get<1>(cell_state);
                Matrix y_t = std::get<2>(cell_state);
                cacheTuple cache_t = std::get<3>(cell_state);

                //Store the new values of the hidden, candidate/memory, and prediction states for the next timestep
                //Matrix a_temp(m, std::vector<double>(n_x));
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        hidden_state[i][timestep][j] = a_next[i][j];
                        prediction[i][timestep][j] = y_t[i][j];
                        candidate[i][timestep][j] = c_next[i][j];
                    }
                }
                cache.push_back(cache_t);
            }

            //Return cache and x-data for backprop
            return std::make_tuple(hidden_state, prediction, candidate, std::make_tuple(cache, x));
        }

    gradientDict lstm_backprop(Tensor3D da, std::tuple<std::vector<cacheTuple>, Tensor3D> fwd_prop_cache) {
            std::vector<cacheTuple> cache = std::get<0>(fwd_prop_cache);
            Tensor3D x = std::get<1>(fwd_prop_cache); // Input

            //Initialize gradients and sizes
            const int m = da.size(), T_x = da[0].size(), n_a = da[0][0].size();
            const int n_x = x[0][0].size();

            Tensor3D dx = linalg::generateZeros(m, T_x, n_x);
            Matrix da_initial = linalg::generateZeros(m, n_a);
            Matrix da_prev_t = linalg::generateZeros(m, n_a);
            Matrix dc_prev_t = linalg::generateZeros(m, n_a);
            Matrix dWf = linalg::generateZeros(n_a, n_a+n_x);
            Matrix dWi = linalg::generateZeros(n_a, n_a+n_x);
            Matrix dWc = linalg::generateZeros(n_a, n_a+n_x);
            Matrix dWo = linalg::generateZeros(n_a, n_a+n_x);
            Matrix dbf = linalg::generateZeros(n_a, 1);
            Matrix dbi = linalg::generateZeros(n_a, 1);
            Matrix dbc = linalg::generateZeros(n_a, 1);
            Matrix dbo = linalg::generateZeros(n_a, 1);

            //Initialize gradients variable
            gradientDict gradients;

            //Backprop iteration through each timestep cell
            for (size_t timestep = T_x; timestep > 0; timestep++) {
                //Compute gradients for each timestep cell
                //Slice the activation data:
                Matrix da_t(m, std::vector<double>(n_a));
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        da_t[i][j] = x[i][timestep][j];
                    }
                }
                //Get the cache tuple for the current timestep
                cacheTuple cache_t = cache.at(timestep);

                //Compute gradients for the current timestep cell
                gradients = LSTMCell::lstm_cell_backward(linalg::add(da_t, da_prev_t), dc_prev_t, cache_t);

                //Store the dx gradient
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        dx[i][timestep][j] = std::get<Matrix>(gradients["dxt"])[i][j];
                    }
                }

                //Add the gradient to the parameter's previous timestep gradients
                dWf = linalg::add(std::get<Matrix>(gradients["dWf"]), dWf);
                dWi = linalg::add(std::get<Matrix>(gradients["dWi"]), dWi);
                dWc = linalg::add(std::get<Matrix>(gradients["dWc"]), dWc);
                dWo = linalg::add(std::get<Matrix>(gradients["dWo"]), dWo);
                dbf = linalg::add(std::get<Matrix>(gradients["dbf"]), dbf);
                dbi = linalg::add(std::get<Matrix>(gradients["dbi"]), dbi);
                dbc = linalg::add(std::get<Matrix>(gradients["dbc"]), dbc);
                dbo = linalg::add(std::get<Matrix>(gradients["dbo"]), dbo);
            }

            // Set the first activation's gradient to backpropagated da_prev gradient
            da_initial = std::get<Matrix>(gradients["da_prev"]);

            gradients["dx"] = dx;
            gradients["da0"] = da_initial;
            gradients["dWf"] = dWf;
            gradients["dbf"] = dbf;
            gradients["dWi"] = dWi;
            gradients["dbi"] = dbi;
            gradients["dWc"] = dWc;
            gradients["dbc"] = dbc;
            gradients["dWo"] = dWo;
            gradients["dbo"] = dbo;

            return gradients;
    }
};