#include "NuralNetwork.h"




NuralNetwork::NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                           double learningRate) : _numInputs(numInputs), _numOutputs(numOutputs), _numHiddenLayers(numHiddenLayers), _numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), _learningRate(learningRate) {
    _layers = std::vector<NeuronLayer>(_numHiddenLayers + 2, NeuronLayer());
}

Eigen::Vector<double, Eigen::Dynamic> NuralNetwork::FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs) {
    Eigen::Vector<double, Eigen::Dynamic> outputs = inputs;
    for (NeuronLayer& layer : _layers) {
        outputs = layer.CalcOutputs(outputs);
    }
    return outputs;
}

void NuralNetwork::BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
    const Eigen::Vector<double, Eigen::Dynamic>& targets) {
    // Calculate the output layer's error
    Eigen::Vector<double, Eigen::Dynamic> outputs = FeedForward(inputs);
    Eigen::Vector<double, Eigen::Dynamic> outputErrors = targets - outputs;
    double meanSquareError = outputErrors.squaredNorm() / _numOutputs;

    // Calculate the output layer's delta
    Eigen::Vector<double, Eigen::Dynamic> outputDeltas = outputErrors.cwiseProduct(outputs.unaryExpr([](double output) {
        return ActivationFunctionDerivative(output, SIGMOID_FUNCTION);
    }));

    // Update the output layer's weights and biases
    _layers.back()._weights += _learningRate * outputDeltas * _layers[_numHiddenLayers - 1].GetOutputs().transpose();
    _layers.back()._biases += _learningRate * outputDeltas;
    
}

