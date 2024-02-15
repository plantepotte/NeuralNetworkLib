#include "NuralNetwork.h"

#include <iostream>

NuralNetwork::NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                           double learningRate) : _numInputs(numInputs), _numOutputs(numOutputs), _numHiddenLayers(numHiddenLayers), _numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), _learningRate(learningRate) {
    _layers = std::vector<NeuronLayer>();
    _layers.reserve(numHiddenLayers + 2);

    // Add the input layer
    _layers.emplace_back(numInputs, numInputs);

    // Add the hidden layers
    for (int i = 0; i < numHiddenLayers; ++i) {
        _layers.emplace_back(numNeuronsPerHiddenLayer, _layers.back().numNeurons);
    }

    // Add the output layer
    _layers.emplace_back(numOutputs, _layers.back().numNeurons);
}

Eigen::Vector<double, Eigen::Dynamic> NuralNetwork::FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs) {
    Eigen::Vector<double, Eigen::Dynamic> outputs = inputs;
    for (NeuronLayer& layer : _layers) {
        outputs = layer.CalcOutputs(outputs);
    }
    return outputs;
}

double NuralNetwork::BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
    const Eigen::Vector<double, Eigen::Dynamic>& targets) {
    // Calculate the output layer's error
    const Eigen::Vector<double, Eigen::Dynamic> outputs = FeedForward(inputs);
    const Eigen::Vector<double, Eigen::Dynamic> outputErrors = targets - outputs;
    std::cout << "Output Errors: " << outputErrors << '\n';
    for (const auto output : outputs) {
        std::cout << "Output: " << output << '\n';
    }
    const double meanSquareError = outputErrors.squaredNorm() / _numOutputs;
    
    _outputDeltas = Eigen::Vector<double, Eigen::Dynamic>(_numOutputs);
    _hiddenDeltas = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(_numHiddenLayers, _numNeuronsPerHiddenLayer);
    
    for (int i = 0; i < _outputDeltas.size(); ++i) {
        _outputDeltas[i] = outputErrors[i] * ActivationLib::ActivationFunctionDerivative(outputs[i], outputActivationFunction);
        _hiddenDeltas(_hiddenDeltas.rows()-1, i) = _layers.back().Weights().row(i).sum() * _outputDeltas[i] * ActivationLib::ActivationFunctionDerivative(_layers[_numHiddenLayers+1].Outputs()[i], hiddenActivationFunction);
    }

    for (int i = _numHiddenLayers - 2; i >= 1; --i) {
        NeuronLayer& layer = _layers[i + 1];
        NeuronLayer& prevLayer = _layers[i];
        for (int j = 0; j < layer.numNeurons; ++j) {
            double error{};
            for (int k = 0; k < layer.numNeuronInputs; ++k) {
                error += layer.Weights()(j, k) * _hiddenDeltas(i + 1, k);
            }
            _hiddenDeltas(i, j) = error * ActivationLib::ActivationFunctionDerivative(prevLayer.Inputs()[j], hiddenActivationFunction);
        }
    }

    _inputDeltas = Eigen::Vector<double, Eigen::Dynamic>(_numInputs);
    for (int i = 0; i < _numInputs; ++i) {
        double error = 0;
        for (int j = 0; j < _numNeuronsPerHiddenLayer; ++j) {
            error += _layers[1].Weights()(j, i) * _hiddenDeltas(0, j);
        }
        _inputDeltas[i] = error * ActivationLib::ActivationFunctionDerivative(_layers[0].Inputs()[i], inputActivationFunction);
    }

    // Update the weights and biases of output layer
    for (int i = 0; i < _layers.back().numNeurons; ++i) {
        for (int j = 0; j < _layers[_numHiddenLayers].numNeuronInputs; ++j) {
            _layers[_numHiddenLayers].Weights()(i, j) += _outputDeltas[i] * _layers[_numHiddenLayers - 1].Inputs()[j] * _learningRate;
        }
        _layers[_numHiddenLayers].Biases()[i] += _outputDeltas[i] * _learningRate;
    }

    // Update the weights and biases of hidden layers
    for (int i = static_cast<int>(_layers.size()) - 1; i >= 2; --i) {
        for (int j = 0; j < _layers[i].numNeurons; ++j) {
            for (int k = 0; k < _layers[i].numNeuronInputs; ++k) {
                _layers[i].Weights()(j, k) += _hiddenDeltas(i-2, j) * _layers[i - 1].Inputs()[k] * _learningRate;
            }
            _layers[i].Biases()[j] += _hiddenDeltas(i-2, j) * _learningRate;
        }
    }

    // Update the weights and biases of input layer
    for (int i = 0; i < _layers[0].numNeurons; ++i) {
        for (int j = 0; j < _numInputs; ++j) {
            _layers[0].Weights()(i, j) += _inputDeltas[i] * inputs[j] * _learningRate;
        }
        _layers[0].Biases()[i] += _inputDeltas[i] * _learningRate;
    }
    
    return meanSquareError;
}

std::string NuralNetwork::Train(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>> targets, int numEpochs) {
    std::string result;
    
    for (int i = 0; i < numEpochs; ++i) {
        double meanSquareError = 0;
        for (int j = 0; j < inputs.size(); ++j) {
            Eigen::Vector<double, Eigen::Dynamic> input = Eigen::Vector<double, Eigen::Dynamic>(inputs[j].size());
            for (int k = 0; k < inputs[j].size(); ++k) {
                input[k] = inputs[j][k];
            }
            Eigen::Vector<double, Eigen::Dynamic> target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < targets[j].size(); ++k) {
                target[k] = targets[j][k];
            }
            meanSquareError += BackPropagate(input, target);
        }
        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError/numEpochs) + "\n";
    }
    
    return result;
}

