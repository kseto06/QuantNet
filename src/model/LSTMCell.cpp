#include "linalg.cpp"
#include "activations.cpp"
#include <vector>
#include <map>

class LSTMCell {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::map<std::string, Matrix> matrixDict;
    
    public:
        linalg np;
        activations act;

        std::tuple< Matrix, Matrix, Matrix, std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> >
        lstm_cell_forward(const Matrix& x_t, const Matrix& a_prev, const Matrix& c_prev, matrixDict& params) {
            /* Inputs:
             * - x_t: current x-input timestep
             * - a_prev: hidden/activation state in the previous timestep
             * - c_prev: memory state at previous timestep
             * - params (matrixDict):
             *      - Wf & bf, weights and biases for the forget gate
             *      - Wu & bu, weights and biases for the update gate
             *      - Wc & bc, weights and biases for the first "tanh" activation
             *      - Wo & bo, weights and biases for the output gate
             *      - Wy & by, weights and biases to relate the hidden-state to the output
             *
             * Outputs:
             * - a_next = matrix, next hidden (activation) state
             * - c_next = matrix, next memory cell state
             * - y_t_pred = the prediction at the current timestep
             * - cache = cached values for backprop -- tuple(a_next, c_next, a_prev, c_prev, x_t, parameters)
             */

            // Get the parameters from params
            Matrix Wf = params["Wf"]; //Forget gates
            Matrix Bf = params["bf"];
            Matrix Wi = params["Wi"]; //Update gates
            Matrix Bi = params["Bi"];
            Matrix Wc = params["Wc"]; //Candidate/memory gates
            Matrix Bc = params["Bc"];
            Matrix Wo = params["Wo"]; //Output gates
            Matrix Bo = params["Bo"];
            Matrix Wy = params["Wy"]; //Prediction weights
            Matrix By = params["By"];

            //Get the dimensions of shapes x_t, W_y
            const int M = x_t.size(), N_X = x_t[0].size(); //Batch size, activations
            const int N_A = Wy.size(), N_Y = Wy[0].size(); //Batch size, input features

            //Concatenate activation/hidden state of the previous state and the current input x_t
            Matrix concat = np.generateZeros(M, N_X+N_A);
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N_A; j++) {
                    concat[i][j] = a_prev[i][j];
                }
            }
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N_X; j++) {
                    concat[i][N_A + j] = x_t[i][j];
                }
            }

            //Compute the forward pass activations using LSTM formulas:
            Matrix candidate = act.tanh(np.add(np.matmul(Wi, concat), Bc));
            Matrix update_gate = act.sigmoid(np.add(np.matmul(concat, Wi), Bi));
            Matrix forget_gate = act.sigmoid(np.add(np.matmul(concat, Wf), Bf));
            Matrix output_gate = act.sigmoid(np.add(np.matmul(concat, Wo), Bo));
            Matrix c_next = np.add(np.elementMultiply(update_gate, candidate), np.elementMultiply(forget_gate, c_prev));
            Matrix a_next = np.elementMultiply(output_gate, act.tanh(c_next));

            //Compute the prediction of the LSTM Cell:
            Matrix yt_pred = act.linear(np.add(np.matmul(Wy, a_next), By));

            //Return next cell parameters and cached values for backprop
            auto params_tuple = std::make_tuple(a_next, c_next, a_prev, c_prev, forget_gate, update_gate, candidate, output_gate, x_t, params);
            return std::make_tuple(a_next, c_next, yt_pred, params_tuple);
        }


};
