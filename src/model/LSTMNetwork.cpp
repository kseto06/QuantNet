#include <vector>
#include <map>
#include "linalg.cpp"
#include "activations.cpp"
#include "LSTMCell.cpp"

class LSTMNetwork {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::map<std::string, Matrix> matrixDict;
    LSTMCell cell;

    public:
        matrixDict initialize_parameters(const int n_input, const int n_hidden, const int n_output) {
            //NOTE: n represents the columns / num of features in the data
            matrixDict params;

            //Initialize parameters to have small values
            //NOTE: We might need to transpose all these values
            //Forget gate:
            params["Wf"] = linalg::scalarMultiply(0.1, linalg::randn(n_input, n_hidden+n_input));
            params["bf"] = linalg::generateZeros(n_input, 1);

            //Update (input) gate:
            params["Wi"] = linalg::scalarMultiply(0.1, linalg::randn(n_hidden, n_hidden+n_input));
            params["bi"] = linalg::generateZeros(n_input, 1);

            //Candidate/memory cells
            params["Wc"] = linalg::scalarMultiply(0.1, linalg::randn(n_hidden, n_hidden+n_input));
            params["bc"] = linalg::generateZeros(n_input, 1);

            //Output gate:
            params["Wo"] = linalg::scalarMultiply(0.1, linalg::randn(n_hidden, n_hidden+n_input));
            params["bo"] = linalg::generateZeros(n_input, 1);

            //Predictions
            params["Wy"] = linalg::scalarMultiply(0.1, linalg::randn(n_output, n_input));
            params["by"] = linalg::generateZeros(n_output, 1);

            return params;
        }

        //Iterate through each cell at their respective timesteps
        std::tuple<Tensor3D, Tensor3D, Tensor3D, std::tuple<std::vector<std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict>>, Tensor3D>>
        lstm_forward(const Tensor3D& x, const Matrix& a_initial, matrixDict& params) {
            /* Inputs:
             * - x: input data, 3D Tensor of shape (num exs, num feats, timestep (days))
             * - a_initial: Initial hidden state
             * - parameters: map of weights and biases
             */

            //NOTE: if hybrid with MLP, cache may not be empty.
            std::vector < std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> > cache;

            Matrix Wy = params["Wy"]; //Get the weight matrix for the prediction

            //Init shapes. NOTE: n_a, n_y might need to be reversed
            const int m = x.size(), n_x = x[0].size(), timesteps = x[0][0].size(), n_y = Wy.size(), n_a = Wy[0].size();

            // Init states
            Tensor3D hidden_state = linalg::generateZeros(m, n_a, timesteps);
            Tensor3D candidate = linalg::generateZeros(m, n_a, timesteps);
            Tensor3D prediction = linalg::generateZeros(m, n_y, timesteps);

            // Init matrices for hidden states at timesteps
            Matrix a_next = a_initial;
            Matrix c_next = linalg::generateZeros(m, n_a);

            for (size_t timestep = 0; timestep < timesteps; timestep++) {
                // Slice the input data at the specific timestep:
                Matrix x_t(m, std::vector<double>(n_x));
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        x_t[i][j] = x[i][j][timestep];
                    }
                }

                //Compute the matrices and parameters for the current timestep cell
                std::tuple< Matrix, Matrix, Matrix, std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> >
                cell_state = cell.lstm_cell_forward(x_t, a_next, c_next, params);

                //Extract the values of the current timestep cell
                a_next = std::get<0>(cell_state), c_next = std::get<1>(cell_state);
                Matrix y_t = std::get<2>(cell_state);
                std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cache_t = std::get<3>(cell_state);

                //Store the new values of the hidden, candidate/memory, and prediction states for the next timestep
                //Matrix a_temp(m, std::vector<double>(n_x));
                for (size_t i = 0; i < m; i++) {
                    for (size_t j = 0; j < n_x; j++) {
                        hidden_state[i][j][timestep] = a_next[i][j];
                        prediction[i][j][timestep] = y_t[i][j];
                        candidate[i][j][timestep] = c_next[i][j];
                    }
                }
                cache.push_back(cache_t);
            }

            //Return cache and x-data for backprop
            return std::make_tuple(hidden_state, prediction, candidate, std::make_tuple(cache, x));
        }
};