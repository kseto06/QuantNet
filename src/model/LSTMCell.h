#ifndef LSTMCELL_H
#define LSTMCELL_H

#include <vector>
#include <map>
#include <variant>
#include <valarray>

namespace LSTMCell {
    //Type definitions
    typedef std::vector<std::vector<double>> Matrix;
    typedef std::vector<std::vector<std::vector<double>>> Tensor3D;
    typedef std::map<std::string, Matrix> matrixDict;

    typedef std::tuple< Matrix, Matrix, Matrix, std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> > forwardTuple;
    typedef std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, matrixDict> cacheTuple;

    typedef std::variant<Matrix, Tensor3D> variantTensor;
    typedef std::map<std::string, variantTensor> gradientDict;

    //Function declarations
    forwardTuple lstm_cell_forward(const Matrix& x_t, const Matrix& a_prev, const Matrix& c_prev, matrixDict& params);
    gradientDict lstm_cell_backward(const Matrix& da_next, const Matrix& dc_next, const cacheTuple& cache);
}

#endif //LSTMCELL_H
