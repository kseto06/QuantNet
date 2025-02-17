<h1 align= "center">C++ LSTM</h1>

<p align="center">This repository contains an implementation of LSTM and MLP from scratch in C++ with zero external dependencies, using only the C++ standard library. 
</p>

<p align="center">This framework supports neural network architectures combining hybrid LSTM and MLP neural networks which allow for a variety of sequence model or time-series applications.
</p>

<br>

Sample implementation (from `src/train_model.cpp`)
```cpp
int main() {
    // Generate sample data
    const int batch_size = 5;
    const int numUnits = 8;
    HybridModel::Tensor3D X_train = linalg::randn(10, 5, 3);  // 10 samples, 5 timesteps, 3 features
    const HybridModel::Matrix Y_train = linalg::randn(10, 1);  // 10 samples, 2 output classes

    //Init data and parameters for HybridModel
    HybridModel::init_data(X_train, Y_train, batch_size);
    HybridModel::init_hidden_units(numUnits);

    // Init model parameters
    const std::vector<std::string> layer_types = {"LSTM", "LSTM", "Relu", "Linear"}; //Neural network
    const std::vector<int> layer_dims = {static_cast<int>(X_train[0][0].size()), 12, 8, static_cast<int>(Y_train.size())}; //Neural network layers/features

    // Initialize the layers
    HybridModel::init_layers(layer_types, layer_dims);

    // Initialize the network parameters
    HybridModel::initialize_network();

    // Generate minibatches
    auto minibatches = HybridModel::generate_minibatches(X_train, Y_train, batch_size, 42);  // Batch size: 2, seed: 42

    // Model iteration through minibatches
    for (const auto& batch : minibatches) {
        auto& X_batch = std::get<0>(batch);
        auto& Y_batch = std::get<1>(batch); 

        // Forward prop
        HybridModel::forward_prop();
        std::cout << "Forward prop done" << std::endl;

        // Compute loss
        HybridModel::loss();
        std::cout << "Loss computed" << std::endl;

        // Backward prop
        HybridModel::back_prop();
        std::cout << "Backprop done.\n";
    }

    return 0;
}
```
<br>
<br>

⚠ **DISCLAIMER: This project is incomplete.** 
<br>
When I initially started this project in November 2024, when I was still wanting to learn more about deep learning
and the different types of models that are possible, I decided to learn more about sequence models and time-series prediction. As such, I learned 
more about LSTMs and I was looking to implement it myself from scratch using C++, to teach myself about LSTMs and the C++ programming language. However, after working on it for 
a long time, I have decided that I wanted to work with more applied ML rather than theoretical implementations, and I wanted to shift my focus towards my 
current big interest (as of Feb. 2025), which is reinforcement learning (RL). 

<br>

Even though it is incomplete, I have still been able to successfully implement forward and backward functions of an LSTM as outlined in the [paper](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf).
I believe that with this, I have already achieved my original goal of being able to implement the LSTM from scratch in C++. And not to mention -- I've already been able to implement a MLP from scratch in Python
in my [(OtakuNet)](https://github.com/kseto06/OtakuNet) project.

<br> 
In the future, I might come back to finish the rest of this project -- we'll see!

## Current Features
- [x] Neural Network Framework (inspired by PyTorch)
  - [x] LSTM Network
    - [x] LSTM Cell Forward & Backward
    - [x] LSTM Forward & Backward
  - [x] MLP Forward & Backward
  - [ ] Optimizers

- [x] Linear Algebra Framework
  - [x] Made from scratch with functionalities inspired by NumPy
     
- [x] Activation Functions
  - [x] ReLU
  - [x] Sigmoid
  - [x] tanh
  - [x] Linear
  
