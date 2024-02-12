#include "NuralNetwork.h"

#include <iostream>

#include "NeuronLayer.h"

double NuralNetwork::ActivationFunction(const double x, const EActivationFunction activationFunction) {
    switch (activationFunction) {
        case HEAVISIDE_STEP_FUNCTION:
            return HeavisideStepFunction(x);
        case SIGMOID_FUNCTION:
            return SigmoidFunction(x);
        case HYPERBOLIC_TANGENT_FUNCTION:
            return HyperbolicTangentFunction(x);
    }
    throw std::invalid_argument("Invalid activation function");
}

double NuralNetwork::ActivationFunctionDerivative(const double x, const EActivationFunction activationFunction) {
    switch (activationFunction) {
        case HEAVISIDE_STEP_FUNCTION:
            return HeavisideStepFunctionDerivative(x);
        case SIGMOID_FUNCTION:
            return SigmoidFunctionDerivative(x);
        case HYPERBOLIC_TANGENT_FUNCTION:
            return HyperbolicTangentFunctionDerivative(x);
    }
    throw std::invalid_argument("Invalid activation function");
}

NuralNetwork::NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                           double learningRate) : _numInputs(numInputs), _numOutputs(numOutputs), _numHiddenLayers(numHiddenLayers), _numNeuronsPerHiddenLayer(numNeuronsPerHiddenLayer), _learningRate(learningRate) {
    _layers = std::vector<NeuronLayer>(_numHiddenLayers + 2, NeuronLayer());
}
