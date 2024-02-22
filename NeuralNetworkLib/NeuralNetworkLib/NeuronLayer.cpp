// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuronLayer.cpp
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 15/2/2024
// //Last Modified On : 22/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#include "NeuronLayer.h"

#include <random>


void NeuronLayer::CalcOutputs() {
    // in-place calculation of the outputs using matrix multiplication and vector addition
    outputs = weights * inputs + biases;
}

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons),
                                                                         numNeuronInputs(numberOfNeuronInputs) {
    // create a random number generator and a uniform distribution between -1 and 1
    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_real_distribution<double> distribution{-1, 1};

    // Initialize the outputs, weights and biases with zeros, random values and random values, respectively
    outputs = Eigen::VectorXd::Zero(numNeurons);
    weights = Eigen::MatrixXd::NullaryExpr(numNeurons, numNeuronInputs, [&distribution, &gen]()
    {
        return distribution(gen);
    });
    biases = Eigen::VectorXd::NullaryExpr(numNeurons, [&distribution, &gen]() { return distribution(gen); });
}

Eigen::Vector<double, Eigen::Dynamic> NeuronLayer::CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& Inputs,
                                                               EActivationFunction activationFunction) {
    inputs = Inputs; // store the inputs

    // in-place calculation of the outputs
    CalcOutputs();

    // create a vector to store the activated outputs
    Eigen::Vector<double, Eigen::Dynamic> activatedOutputs = outputs;

    // Apply the activation function to each output
    for (auto& activatedOutput : activatedOutputs) {
        activatedOutput = ActivationLib::ActivationFunction(activatedOutput, activationFunction);
    }
    return activatedOutputs;
}
