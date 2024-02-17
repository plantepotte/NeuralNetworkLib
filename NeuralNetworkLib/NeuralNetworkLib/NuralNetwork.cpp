#include "NuralNetwork.h"

#include <iostream>

NuralNetwork::NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                           double learningRate) : _numInputs(numInputs), _numOutputs(numOutputs), _numHiddenLayers(numHiddenLayers), _numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), _learningRate(learningRate) {
    _layers = std::vector<NeuronLayer>();
    _layers.reserve(_numHiddenLayers + 2);

    // Add the input layer
    _layers.emplace_back(_numInputs, _numInputs);

    // Add the hidden layers
    for (int i = 0; i < _numHiddenLayers; ++i) {
        _layers.emplace_back(_numNeuronsPerHiddenLayer, _layers.back().numNeurons);
    }

    // Add the output layer
    _layers.emplace_back(_numOutputs, _layers.back().numNeurons);
}

Eigen::Vector<double, Eigen::Dynamic> NuralNetwork::FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs) {
    Eigen::Vector<double, Eigen::Dynamic> outputs = inputs;
    for (NeuronLayer& layer : _layers) {
        outputs = layer.CalcOutputs(outputs);
        // std::cout << "Outputs: " << outputs << '\n';
    }
    return outputs;
}

double NuralNetwork::BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
    const Eigen::Vector<double, Eigen::Dynamic>& targets) {

    const Eigen::Vector<double, Eigen::Dynamic> outputs = FeedForward(inputs);
    Eigen::Vector<double, Eigen::Dynamic> outputErrors = targets - outputs;

    const double meanSquareError = outputErrors.squaredNorm() / _numOutputs;
    
    _neuronDeltas = std::vector<Eigen::Vector<double, Eigen::Dynamic>>(_layers.size());
    for (int i = 0; i < _neuronDeltas.size(); ++i) {
        _neuronDeltas[i] = Eigen::Vector<double, Eigen::Dynamic>(_layers[i].numNeurons);
    }
    
    const auto numLayers = static_cast<int>(_layers.size());
    for (int i = numLayers - 1; i >= 0; --i) {
        if (i >= numLayers - 1) {
            for (int j = 0; j < _numOutputs; ++j) {
                _neuronDeltas.back()[j] = outputErrors[j] * ActivationLib::ActivationFunctionDerivative(_layers.back().outputs[j], outputActivationFunction);
            }
        }
        else if (i <= 0) {
            _neuronDeltas[i] = _layers[i+1].weights.transpose() * _neuronDeltas[i+1];
            for (int j = 0; j < _neuronDeltas[i].size(); ++j) {
                _neuronDeltas[i][j] *= ActivationLib::ActivationFunctionDerivative(_layers[i].outputs[j], inputActivationFunction);
            }
        }
        else {
            _neuronDeltas[i] = _layers[i+1].weights.transpose() * _neuronDeltas[i+1];
            for (int j = 0; j < _neuronDeltas[i].size(); ++j) {
                _neuronDeltas[i][j] *= ActivationLib::ActivationFunctionDerivative(_layers[i].outputs[j], hiddenActivationFunction);
            }
        }
        _layers[i].weights += _learningRate * _neuronDeltas[i] * _layers[i].inputs.transpose();
        _layers[i].biases += _learningRate * _neuronDeltas[i];
        
    }
    
    return meanSquareError;
}

std::string NuralNetwork::Train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, const int numEpochs) {
    std::string result;
    
    for (int i = 0; i < numEpochs; ++i) {
        double meanSquareError = 0;
        for (int j = 0; j < inputs.size(); ++j) {
            auto input = Eigen::Vector<double, Eigen::Dynamic>(inputs[j].size());
            for (int k = 0; k < inputs[j].size(); ++k) {
                input[k] = inputs[j][k];
            }
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < targets[j].size(); ++k) {
                target[k] = targets[j][k];
            }
            meanSquareError += BackPropagate(input, target);
        }
        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError) + "\n";
    }
    
    return result;
}

std::string NuralNetwork::Train(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, double maxError, int maxEpochs) {

    std::string result{};
    double meanSquareError{std::numeric_limits<double>::max()};
    int i{};
    
    while (meanSquareError > maxError && maxEpochs > i++) {
        meanSquareError = 0;
        
        for (int j = 0; j < inputs.size(); ++j) {
            auto input = Eigen::Vector<double, Eigen::Dynamic>(inputs[j].size());
            for (int k = 0; k < inputs[j].size(); ++k) {
                input[k] = inputs[j][k];
            }
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < targets[j].size(); ++k) {
                target[k] = targets[j][k];
            }
            meanSquareError += BackPropagate(input, target);
        }
        
        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError) + "\n";
    }
    
    return result;
}

