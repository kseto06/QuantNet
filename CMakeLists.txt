cmake_minimum_required(VERSION 3.29)
project(StockPredictionApp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS 0FF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O0")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Enabling AddressSanitizer for Debug build")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

add_executable(StockPredictionApp
        QuantNet/src/model/linalg.cpp
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
