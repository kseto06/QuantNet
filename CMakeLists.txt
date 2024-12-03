cmake_minimum_required(VERSION 3.29)
project(StockPredictionApp)

set(CMAKE_CXX_STANDARD 20)

add_executable(StockPredictionApp
        QuantNet/src/model/linalg.cpp
        QuantNet/src/model/MultilayerPerceptron.cpp
        QuantNet/src/train_model.cpp
        QuantNet/src/model/activations.cpp
        QuantNet/src/model/LSTMCell.cpp
        QuantNet/src/model/LSTMNetwork.cpp
        QuantNet/src/model/linalg.h
        QuantNet/src/model/activations.h
        QuantNet/src/model/LSTMCell.h
        QuantNet/src/model/LSTMNetwork.h
        QuantNet/src/model/MLP.cpp
        QuantNet/src/model/MLP.h
        QuantNet/src/model/HybridModel.cpp
        QuantNet/src/model/HybridModel.h
)
