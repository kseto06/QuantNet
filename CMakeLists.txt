cmake_minimum_required(VERSION 3.29)
project(cpp-lstm VERSION 0.1.0 LANGUAGES CXX)

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

add_executable(cpp-lstm
        src/model/linalg.cpp
        src/train_model.cpp
        src/model/activations.cpp
        src/model/LSTMCell.cpp
        src/model/LSTMNetwork.cpp
        src/model/linalg.h
        src/model/activations.h
        src/model/LSTMCell.h
        src/model/LSTMNetwork.h
        src/model/MLP.cpp
        src/model/MLP.h
        src/model/HybridModel.cpp
        src/model/HybridModel.h
)

set_property(TARGET cpp-lstm PROPERTY CXX_STANDARD 20)
