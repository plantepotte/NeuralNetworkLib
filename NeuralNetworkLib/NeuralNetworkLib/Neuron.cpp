// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: Perceptron.cpp
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 12/2/2024
// //Last Modified On : 12/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#include "Neuron.h"

#include <random>

double Neuron::BiasedDotProd(const std::vector<double>& weights, const std::vector<double>& inputs,
                                 const double bias) {
    if (weights.size() != inputs.size()) { throw std::invalid_argument("Weights and inputs must be of the same size"); }

    double dotProd{};
    for (size_t i = 0; i < weights.size(); ++i) { dotProd += weights[i] * inputs[i]; }
    return dotProd + bias;
}

std::string Neuron::TrainEpoch(int j) {
    std::string status = "Epoch " + std::to_string(j) + " - \n";
    _rmsError = 0;

    for (auto element : _ts) {
        // Calculate the output of the perceptron
        const auto output = StepFunction(BiasedDotProd(_weights, element.input, _bias));
        const auto error = element.output - output;

        status += "Input: " + std::to_string(element.input[0]) + ", " + std::to_string(element.input[1]) + " -> " +
            std::to_string(element.output) + ", Result: " + std::to_string(output) + ", ";
        status += "Error: " + std::to_string(error) + "\n";
        status += "Weights: ";
        for (const auto weight : _weights) { status += std::to_string(weight) + ", "; }
        status += "Bias: " + std::to_string(_bias) + "\n";

        _rmsError += error * error;
        for (size_t i = 0; i < _weights.size(); ++i) { _weights[i] += error * element.input[i]; }
        _bias += error;
    }
    if (!_ts.empty()) {
        // use root mean square error
        _rmsError /= _ts.size();
        _rmsError = std::sqrt(_rmsError);
    }

    status += "Root mean square error: " + std::to_string(_rmsError) + "\n";
    return status;
}

Neuron::Neuron(int numNeuronInputs) {
}

std::string Neuron::Train(int epochs) {
    std::string status{};
    for (int j = 0; j < epochs; ++j) { status += TrainEpoch(j); }
    return status;
}

std::string Neuron::Train(double tolerance, int epochLimit) {
    std::string status{};
    int j = 0;
    do {
        status += TrainEpoch(j);
        ++j;
    }
    while (_rmsError > tolerance && j < epochLimit);
    status += "Tolerance: " + std::to_string(tolerance) + " reached after " + std::to_string(j) + " epochs\n";
    return status;
}

Neuron::Neuron(std::vector<TrainingSet> ts): _ts(std::move(ts)) {
    // use Mersenne Twister 19937 64bit as the random number generator
    std::mt19937_64 mt{std::random_device{}()};

    // use uniform distribution to generate random numbers between 0 and 1
    std::uniform_real_distribution<double> dist{0.0, 1.0};

    // initialise the weights and bias to random values
    _weights = {dist(mt), dist(mt)};
    _bias = dist(mt);
}
