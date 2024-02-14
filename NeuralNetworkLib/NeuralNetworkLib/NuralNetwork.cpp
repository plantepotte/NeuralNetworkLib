#include "NuralNetwork.h"

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
    const double meanSquareError = outputErrors.squaredNorm() / _numOutputs;
    

    
    return meanSquareError;
}

std::string NuralNetwork::Train(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& inputs,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& targets, int numEpochs) {
    std::string result;
    double meanSquareError = 0;
    for (int i = 0; i < numEpochs; ++i) {
        for (int j = 0; j < inputs.cols(); ++j) {
            meanSquareError += BackPropagate(inputs.col(j), targets.col(j));
        }
        result += "Epoch " + std::to_string(i) + " Mean Square Error: " + std::to_string(meanSquareError/numEpochs) + "\n";
    }
    
    return result;
}

