#include "NeuronLayer.h"

#include <iostream>
#include <random>


void NeuronLayer::CalcOutputs() {
    // perform the dot product of the weights and inputs and add the biases
    outputs = weights * inputs + biases;
}

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons), numNeuronInputs(numberOfNeuronInputs) {

    std::random_device rd{};
    std::mt19937 gen(rd());  //here you could also set a seed
    std::normal_distribution<double> dis(0, 0.5);
    
    // Initialize the outputs, weights and biases with zeros, random values and random values, respectively
    outputs = Eigen::VectorXd::Zero(numNeurons);
    weights = Eigen::MatrixXd::NullaryExpr(numNeurons, numNeuronInputs, [&dis, &gen](){ return dis(gen); });
    biases = Eigen::VectorXd::NullaryExpr(numNeurons, [&dis, &gen](){ return dis(gen); });

    // std::cout << "Weights: " << weights << '\n';
    // std::cout << "Biases: " << biases << '\n';
}

Eigen::Vector<double, Eigen::Dynamic> NeuronLayer::CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& Inputs) {
    inputs = Inputs;
    CalcOutputs();
    Eigen::Vector<double, Eigen::Dynamic> activatedOutputs = outputs;
    // Apply the activation function to each output
    for (double& output : activatedOutputs) {
        output = ActivationLib::ActivationFunction(output, activationFunction);
    }
    return activatedOutputs;
}
