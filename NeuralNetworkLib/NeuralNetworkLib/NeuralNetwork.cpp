// //////////////////////////////////////////////////////////////////////////
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
#include <fstream>
#include <iostream>

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
    _layers[i].biases += _learningRate * grad;
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

    // calculate deltas of output layer
    for (int j = 0; j < _numOutputs; ++j) {
        // calculate the neuron deltas of the output layer
        _neuronDeltas.back()[j] = outputErrors[j] * ActivationLib::ActivationFunctionDerivative(
            _layers.back().outputs[j], _outputActivationFunction);
    }
    
    // calculate the neuron deltas of the hidden layers
    for (int i = numLayers - 2; i >= 0; --i) {
        
        // calculate vector of weights * neuronDeltas for subsequent layer
        _neuronDeltas[i] = _layers[i + 1].weights.transpose() * _neuronDeltas[i + 1];

        // multiply gradient sums with the derivative of the activation function
        for (int j = 0; j < _neuronDeltas[i].size(); ++j) {
            _neuronDeltas[i][j] *= ActivationLib::ActivationFunctionDerivative(
                _layers[i].outputs[j], _hiddenActivationFunction);
        }
    }

    // update the weights and biases
    for (int i = 0; i < numLayers-1; ++i) {
        UpdateWeightsAndBiases(_neuronDeltas[i], i);
    }
    UpdateWeightsAndBiases(outputErrors, numLayers - 1);

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
            for (int k = 0; k < static_cast<int>(inputs[j].size()); ++k) {
                input[k] = inputs[j][k];
            }
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < static_cast<int>(targets[j].size()); ++k) {
                target[k] = targets[j][k];
            }
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
            for (int k = 0; k < static_cast<int>(inputs[j].size()); ++k) {
                input[k] = inputs[j][k];
            }
            
            auto target = Eigen::Vector<double, Eigen::Dynamic>(targets[j].size());
            for (int k = 0; k < static_cast<int>(targets[j].size()); ++k) {
                target[k] = targets[j][k];
            }
            meanSquareError += BackPropagate(input, target);
        }

        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError) + "\n";
    }

    return result;
}

bool NeuralNetwork::SaveToFile(const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // save the network parameters
        file << _numInputs << " " << _numOutputs << " " << _numHiddenLayers << " " << _numNeuronsPerHiddenLayer << " "
             << _learningRate << " " << static_cast<int>(_hiddenActivationFunction) << " "
             << static_cast<int>(_outputActivationFunction) << "\n";

        // save the layers
        for (const auto& layer : _layers) {
            file << layer.numNeurons << " " << layer.numNeuronInputs << "\n";
            for (int i = 0; i < layer.weights.rows(); ++i) {
                for (int j = 0; j < layer.weights.cols(); ++j) {
                    file << layer.weights(i, j) << " ";
                }
                file << "\n";
            }
            for (const double bias : layer.biases) {
                file << bias << " ";
            }
            file << "\n";
        }
        return true;
    }

    std::cerr << "Could not open file " << filename << '\n';
    return false;
}

bool NeuralNetwork::LoadFromFile(const std::string& filename) {
    std::ifstream file(filename);

    if (file.is_open()) {
        // load the network parameters
        file >> _numInputs >> _numOutputs >> _numHiddenLayers >> _numNeuronsPerHiddenLayer >> _learningRate;
        
        int hiddenActivationFunction, outputActivationFunction;
        file >> hiddenActivationFunction >> outputActivationFunction;
        
        _hiddenActivationFunction = static_cast<EActivationFunction>(hiddenActivationFunction);
        _outputActivationFunction = static_cast<EActivationFunction>(outputActivationFunction);

        _layers.clear();
        _layers.reserve(_numHiddenLayers + 1);

        // load the layers
        for (int i = 0; i < _numHiddenLayers + 1; ++i) {
            int numNeurons, numNeuronInputs;
            file >> numNeurons >> numNeuronInputs;
            _layers.emplace_back(numNeurons, numNeuronInputs);
            for (int j = 0; j < _layers[i].weights.rows(); ++j) {
                for (int k = 0; k < _layers[i].weights.cols(); ++k) {
                    file >> _layers[i].weights(j, k);
                }
            }
            for (double& bias : _layers[i].biases) {
                file >> bias;
            }
        }
        return true;
    }

    std::cerr << "Could not open file " << filename << '\n';
    return false;
}
