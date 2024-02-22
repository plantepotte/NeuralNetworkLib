﻿// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuralNetwork.cpp
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 22/2/2024
// //Last Modified On : 22/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                             double learningRate) : _numInputs(numInputs), _numOutputs(numOutputs),
                                                    _numHiddenLayers(numHiddenLayers),
                                                    _numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer),
                                                    _learningRate(learningRate) {
    // reserve space for the layers
    _layers = std::vector<NeuronLayer>();
    _layers.reserve(_numHiddenLayers + 1);

    // Add the first hidden layer
    _layers.emplace_back(_numNeuronsPerHiddenLayer, _numInputs);

    // Add the hidden layers
    for (int i = 0; i < _numHiddenLayers - 1; ++i) {
        _layers.emplace_back(_numNeuronsPerHiddenLayer, _layers.back().numNeurons);
    }

    // Add the output layer
    _layers.emplace_back(_numOutputs, _layers.back().numNeurons);
}

Eigen::Vector<double, Eigen::Dynamic> NeuralNetwork::FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs) {
    // store the inputs
    Eigen::Vector<double, Eigen::Dynamic> outputs = inputs;

    // set the activation function to the hidden layer activation function
    EActivationFunction activationFunction = _hiddenActivationFunction;
    for (int i = 0; i < static_cast<int>(_layers.size()); ++i) {
        // set the activation function to the output layer activation function if the current layer is the output layer
        if (i >= static_cast<int>(_layers.size()) - 1) { activationFunction = _outputActivationFunction; }

        // calculate the outputs of the layer and store them
        outputs = _layers[i].CalcOutputs(outputs, activationFunction);
    }
    // return the final outputs
    return outputs;
}

void NeuralNetwork::UpdateWeightsAndBiases(const Eigen::Vector<double, Eigen::Dynamic>& grad, const int i) {
    // loop through the weights and biases and update them
    for (int row = 0; row < _layers[i].weights.rows(); ++row) {
        for (int col = 0; col < _layers[i].weights.cols(); ++col) {
            _layers[i].weights(row, col) += _learningRate * grad[row] * _layers[i].inputs[col];
        }
    }
    _layers[i].biases += _learningRate * _neuronDeltas[i];
}

double NeuralNetwork::BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
                                    const Eigen::Vector<double, Eigen::Dynamic>& targets) {
    // calculate the outputs of the network and the errors
    const Eigen::Vector<double, Eigen::Dynamic> outputs = FeedForward(inputs);
    Eigen::Vector<double, Eigen::Dynamic> outputErrors = targets - outputs;

    // calculate the mean square error
    const double meanSquareError = 0.5 * outputErrors.squaredNorm() / _numOutputs;

    // reserve space for the neuron deltas
    _neuronDeltas = std::vector<Eigen::Vector<double, Eigen::Dynamic>>(_layers.size());
    for (int i = 0; i < static_cast<int>(_neuronDeltas.size()); ++i) {
        _neuronDeltas[i] = Eigen::Vector<double, Eigen::Dynamic>(_layers[i].numNeurons);
    }

    const auto numLayers = static_cast<int>(_layers.size());
    for (int i = numLayers - 1; i >= 0; --i) {
        if (i >= numLayers - 1) {
            // output layer
            for (int j = 0; j < _numOutputs; ++j) {
                // calculate the neuron deltas of the output layer
                _neuronDeltas.back()[j] = outputErrors[j] * ActivationLib::ActivationFunctionDerivative(
                    _layers.back().outputs[j], _outputActivationFunction);
            }
        }
        else {
            // calculate the neuron deltas of the hidden layers
            _neuronDeltas[i] = _layers[i + 1].weights.transpose() * _neuronDeltas[i + 1];
            for (int j = 0; j < _neuronDeltas[i].size(); ++j) {
                _neuronDeltas[i][j] *= ActivationLib::ActivationFunctionDerivative(
                    _layers[i].outputs[j], _hiddenActivationFunction);
            }
        }
    }

    // update the weights and biases
    for (int i = 0; i < static_cast<int>(_layers.size()); ++i) {
        if (i < 1) { UpdateWeightsAndBiases(outputErrors, i); }
        else { UpdateWeightsAndBiases(_neuronDeltas[i], i); }
    }

    return meanSquareError;
}

std::string NeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets, const int numEpochs) {
    std::string result;

    for (int i = 0; i < numEpochs; ++i) {
        double meanSquareError = 0; // mean square error

        // loop through the inputs and targets and back propagate the error
        for (int j = 0; j < static_cast<int>(inputs.size()); ++j) {
            auto input = Eigen::Vector<double, Eigen::Dynamic>(inputs[j].size());
            for (int k = 0; k < static_cast<int>(inputs[j].size()); ++k) { input[k] = inputs[j][k]; }
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < static_cast<int>(targets[j].size()); ++k) { target[k] = targets[j][k]; }
            meanSquareError += BackPropagate(input, target);
        }
        result += "Epoch " + std::to_string(i) + ", Mean Square Error: " + std::to_string(meanSquareError) + "\n";
    }

    return result;
}

std::string NeuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
                                 const std::vector<std::vector<double>>& targets, const double maxError,
                                 const int maxEpochs) {
    std::string result{};
    double meanSquareError{std::numeric_limits<double>::max()};
    int i{};

    // loop through the inputs and targets and back propagate the error until the error is below the threshold
    while (meanSquareError > maxError && maxEpochs > i++) {
        meanSquareError = 0;

        for (int j = 0; j < static_cast<int>(inputs.size()); ++j) {
            auto input = Eigen::Vector<double, Eigen::Dynamic>(inputs[j].size());
            for (int k = 0; k < static_cast<int>(inputs[j].size()); ++k) { input[k] = inputs[j][k]; }
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < static_cast<int>(targets[j].size()); ++k) { target[k] = targets[j][k]; }
            meanSquareError += BackPropagate(input, target);
        }

        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError) + "\n";
    }

    return result;
}
