#include "NeuronLayer.h"

void NeuronLayer::CalcOutputs() {
    // perform the dot product of the weights and inputs and add the biases
    _outputs = _weights * _inputs + _biases;

    // Apply the activation function to each output
    for (double& output : _outputs) {
        output = ActivationLib::ActivationFunction(output, activationFunction);
    }
}

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons), numNeuronInputs(numberOfNeuronInputs) {

    // Initialize the outputs, weights and biases with zeros, random values and random values, respectively
    _outputs = Eigen::VectorXd::Zero(numNeurons);
    _weights = Eigen::MatrixXd::Random(numNeurons, numNeuronInputs);
    _biases = Eigen::VectorXd::Random(numNeurons);
}

Eigen::Vector<double, Eigen::Dynamic> NeuronLayer::CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& inputs) {
    _inputs = inputs;
    CalcOutputs();
    return _outputs;
}
