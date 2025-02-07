#include "linalg.h"
#include "activations.h"
#include "LSTMCell.h"
#include <vector>
#include <map>
#include <variant>

namespace LSTMCell {
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::map<std::string, Matrix> matrixDict;

    typedef std::tuple< Matrix, Matrix, Matrix, std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> > forwardTuple;
    typedef std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cacheTuple;

    typedef std::variant<Matrix, Tensor3D> variantTensor;
    typedef std::map<std::string, variantTensor> gradientDict;
    typedef std::vector<cacheTuple> forwardCaches;

    forwardTuple lstm_cell_forward(const Matrix& x_t, const Matrix& a_prev, const Matrix& c_prev, matrixDict& params, const int layer) {
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
            Matrix Wf = params["Wf"+std::to_string(layer)]; //Forget gates
            Matrix Bf = params["bf"+std::to_string(layer)];
            Matrix Wi = params["Wi"+std::to_string(layer)]; //Update gates
            Matrix Bi = params["bi"+std::to_string(layer)];
            Matrix Wc = params["Wc"+std::to_string(layer)]; //Candidate/memory gates
            Matrix Bc = params["bc"+std::to_string(layer)];
            Matrix Wo = params["Wo"+std::to_string(layer)]; //Output gates
            Matrix Bo = params["bo"+std::to_string(layer)];
            Matrix Wy = params["Wy"+std::to_string(layer)]; //Prediction weights
            Matrix By = params["by"+std::to_string(layer)];

            //Get the dimensions of shapes x_t, W_y
            const int M = x_t.size(), N_X = x_t[0].size(); //Num of exs, features at current timestep
            const int N_A = Wy[0].size(), N_Y = Wy.size(); //Num of hidden states, num of outputs

            // std::cout << " DEBUG - Checking shapes in LSTMCell Forward" << std::endl;
            // std::cout << "  Shape of x_t: " << linalg::shape(x_t) << std::endl;
            // std::cout << "  Shape of a_prev: " << linalg::shape(a_prev) << std::endl;
            // std::cout << "  Shape of c_prev (input): " << linalg::shape(c_prev) << std::endl;
            // std::cout << "  Shape of Wf: " << linalg::shape(Wf) << std::endl;
            // std::cout << "  Shape of Bf: " << linalg::shape(Bf) << std::endl;
            // std::cout << "  Shape of Wi: " << linalg::shape(Wi) << std::endl;
            // std::cout << "  Shape of Bi: " << linalg::shape(Bi) << std::endl;
            // std::cout << "  Shape of Wc: " << linalg::shape(Wc) << std::endl;
            // std::cout << "  Shape of Bc: " << linalg::shape(Bc) << std::endl;
            // std::cout << "  Shape of Wo: " << linalg::shape(Wo) << std::endl;
            // std::cout << "  Shape of Bo: " << linalg::shape(Bo) << std::endl;

            //Concatenate activation/hidden state of the previous state and the current input x_t
            Matrix concat = linalg::generateZeros(M, N_X+N_A);
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N_X; j++) {
                    concat[i][j] = a_prev[i][j];
                }
            }
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N_X; j++) {
                    concat[i][N_A + j] = x_t[i][j];
                }
            }
            // std::cout << " DEBUG - Shape of concat: " << linalg::shape(concat) << std::endl;

            // std::cout << "LSTM-Cell Forward concat successful" << std::endl;

            //Compute the forward pass activations using LSTM formulas:
            Matrix candidate = activations::tanh(linalg::add(linalg::matmul(Wi, linalg::transpose(concat)), Bc));
            Matrix update_gate = activations::sigmoid(linalg::add(linalg::matmul(Wi, linalg::transpose(concat)), Bi));
            Matrix forget_gate = activations::sigmoid(linalg::add(linalg::matmul(Wf, linalg::transpose(concat)), Bf));
            Matrix output_gate = activations::sigmoid(linalg::add(linalg::matmul(Wo, linalg::transpose(concat)), Bo));
            Matrix c_next = linalg::transpose(linalg::add(linalg::elementMultiply(update_gate, candidate), linalg::transpose(linalg::elementMultiply(linalg::transpose(forget_gate), c_prev))));
            Matrix a_next = linalg::transpose(linalg::elementMultiply(output_gate, linalg::transpose(activations::tanh(c_next))));

            //Compute the prediction of the LSTM Cell:
            Matrix yt_pred = linalg::transpose(activations::linear(linalg::add(linalg::matmul(Wy, a_next), By)));

            //Return next cell parameters and cached values for backprop
            auto params_tuple = std::make_tuple(a_next, c_next, a_prev, c_prev, forget_gate, update_gate, candidate, output_gate, x_t, params);

            return std::make_tuple(a_next, c_next, yt_pred, params_tuple);
    }

    //Compute back propagation for a single LSTM cell
    gradientDict lstm_cell_backward(const Matrix& da_next, const Matrix& dc_next, const cacheTuple& cache) {
            /* Inputs:
             * - da_next, gradients of next hidden state, Matrix (m, n_a)
             * - dc_next, gradients of next candidate/memory state, Matrix (m, n_a)
             * - cache, forward pass tuple
             */
            //Retrieve forward prop information
            Matrix a_next = std::get<0>(cache);
            Matrix c_next = std::get<1>(cache);
            Matrix a_prev = std::get<2>(cache);
            Matrix c_prev = std::get<3>(cache);
            Matrix f_gate = std::get<4>(cache);
            Matrix u_gate = std::get<5>(cache);
            Matrix candidate = std::get<6>(cache);
            Matrix o_gate = std::get<7>(cache);
            Matrix x_t = std::get<8>(cache);
            matrixDict params = std::get<9>(cache);

            //Retrieve shapes
            const int m_x = x_t.size(), m_a = a_next.size(), n_x = x_t[0].size(), n_a = a_next[0].size();

            //Compute gate derivatives
            Matrix do_gate_t = linalg::elementMultiply(
                                linalg::elementMultiply(
                                    linalg::elementMultiply(da_next, activations::tanh(c_next)), o_gate),
                                    linalg::subtract(
                                        linalg::generateOnes(o_gate.size(), o_gate[0].size()), o_gate));

            Matrix dcc_t = linalg::add(
                            linalg::elementMultiply(dc_next, u_gate),
                              linalg::elementMultiply(
                                linalg::elementMultiply(
                                    linalg::elementMultiply(
                                        linalg::elementMultiply(
                                            linalg::elementMultiply(
                                                o_gate,
                                                activations::tanh_prime(c_next)
                                                ),
                                                u_gate
                                            ),
                                            da_next
                                        ),
                                        candidate
                                    ),
                                    activations::tanh_prime(candidate)
                                )
                              );

            Matrix du_gate_t = linalg::elementMultiply(
                                linalg::elementMultiply(
                                    linalg::elementMultiply(
                                        linalg::add(
                                            linalg::elementMultiply(activations::tanh_prime(c_next),
                                                linalg::elementMultiply(
                                                    o_gate, da_next)
                                                ), dc_next),
                                            candidate),
                                        u_gate),
                                    linalg::subtract(
                                        linalg::generateOnes(u_gate.size(), u_gate[0].size()), u_gate));

            Matrix df_gate_t = linalg::elementMultiply(
                                linalg::elementMultiply(
                                    linalg::elementMultiply(
                                        linalg::add(
                                            linalg::elementMultiply(activations::tanh_prime(c_next),
                                                linalg::elementMultiply(
                                                    o_gate, da_next)
                                                ), dc_next),
                                            c_prev),
                                        f_gate),
                                    linalg::subtract(
                                        linalg::generateOnes(f_gate.size(), f_gate[0].size()), f_gate));

            //Concatenate activation/hidden state of the previous state and the input x_t for derivatives of weight gates on axis=0:
            const int concat_cols = std::max(n_a, n_x);
            Matrix concat = linalg::generateZeros(m_a+m_x, concat_cols);

            for (size_t i = 0; i < m_a; i++) {
                for (size_t j = 0; j < concat_cols; j++) {
                    if (j < n_a) {
                        concat[i][j] = a_prev[i][j];
                    }
                }
            }

            // Copy x_t into the bottom part of concat, padding if necessary
            for (size_t i = 0; i < m_x; i++) {
                for (size_t j = 0; j < concat_cols; j++) {
                    if (j < n_x) {
                        concat[m_a + i][j] = x_t[i][j];
                    }
                }
            }

            //Compute parameter derivatives with gate derivatives
            Matrix dWf = linalg::matmul(df_gate_t, linalg::transpose(concat));
            Matrix dWi = linalg::matmul(du_gate_t, linalg::transpose(concat));
            Matrix dWc = linalg::matmul(dcc_t, linalg::transpose(concat));
            Matrix dWo = linalg::matmul(do_gate_t, linalg::transpose(concat));
            Matrix dbf = linalg::sum(df_gate_t, 1);
            Matrix dbi = linalg::sum(du_gate_t, 1);
            Matrix dbc = linalg::sum(dcc_t, 1);
            Matrix dbo = linalg::sum(do_gate_t, 1);

            //Compute the final derivatives of the previous memory and hidden states, and the input
            Matrix da_prev1 = linalg::add(
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wf"], 0, n_a)), df_gate_t),
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wi"], 0, n_a)), du_gate_t));
            Matrix da_prev2 = linalg::add(
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wc"], 0, n_a)), dcc_t),
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wo"], 0, n_a)), do_gate_t));
            Matrix da_prev = linalg::add(da_prev1, da_prev2);

            Matrix dc_prev = linalg::add(
                                linalg::elementMultiply(dc_next, f_gate),
                                linalg::elementMultiply(
                                    linalg::elementMultiply(
                                        linalg::elementMultiply(f_gate, da_next),
                                        activations::tanh_prime(c_next)
                                    ), o_gate)
                                );

            Matrix dx_t1 = linalg::add(
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wf"], n_a, params["Wf"][0].size())), df_gate_t),
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wi"], n_a, params["Wi"][0].size())), du_gate_t));
            Matrix dx_t2 = linalg::add(
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wc"], n_a, params["Wc"][0].size())), dcc_t),
                            linalg::matmul(linalg::transpose(linalg::sliceCols(params["Wo"], n_a, params["Wo"][0].size())), do_gate_t));
            Matrix dx_t = linalg::add(dx_t1, dx_t2);

            gradientDict gradients;
            gradients["dxt"] = dx_t;
            gradients["da_prev"] = da_prev;
            gradients["dc_prev"] = dc_prev;
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
