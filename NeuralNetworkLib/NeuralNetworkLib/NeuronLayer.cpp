#include "NeuronLayer.h"

#include <iostream>
#include <random>


void NeuronLayer::CalcOutputs() {
    // perform the dot product of the weights and inputs and add the biases
    outputs = weights * inputs + biases;
}

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons), numNeuronInputs(numberOfNeuronInputs) {

    std::random_device rd{};
    std::mt19937_64 gen{rd()};  //here you could also set a seed
    std::uniform_real_distribution<double> distribution{-1, 1};
    
    // Initialize the outputs, weights and biases with zeros, random values and random values, respectively
    outputs = Eigen::VectorXd::Zero(numNeurons);
    weights = Eigen::MatrixXd::NullaryExpr(numNeurons, numNeuronInputs, [&distribution, &gen](){ return distribution(gen); });
    biases = Eigen::VectorXd::NullaryExpr(numNeurons, [&distribution, &gen](){ return distribution(gen); });

    // std::cout << "Weights: " << weights << '\n';
    // std::cout << "Biases: " << biases << '\n';
}

Eigen::Vector<double, Eigen::Dynamic> NeuronLayer::CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& Inputs, EActivationFunction activationFunction) {
    inputs = Inputs;
    
    CalcOutputs();
    Eigen::Vector<double, Eigen::Dynamic> activatedOutputs = outputs;

    // Apply the activation function to each output
    for (auto& activatedOutput : activatedOutputs) {
        activatedOutput = ActivationLib::ActivationFunction(activatedOutput, activationFunction);
    }
    return activatedOutputs;
}
