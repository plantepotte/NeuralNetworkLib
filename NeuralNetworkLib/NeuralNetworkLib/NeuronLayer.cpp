#include "NeuronLayer.h"


void NeuronLayer::CalcOutputs() {
    // perform the dot product of the weights and inputs and add the biases
    outputs = weights * inputs + biases;
}

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons), numNeuronInputs(numberOfNeuronInputs) {

    // Initialize the outputs, weights and biases with zeros, random values and random values, respectively
    outputs = Eigen::VectorXd::Zero(numNeurons);
    weights = Eigen::MatrixXd::Random(numNeurons, numNeuronInputs);
    biases = Eigen::VectorXd::Random(numNeurons);
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
