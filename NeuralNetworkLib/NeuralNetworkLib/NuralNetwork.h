// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NuralNetwork.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 12/2/2024
// //Last Modified On : 12/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#pragma once
#include <cmath>
#include <vector>

class NeuronLayer;

class NuralNetwork {
public:
    enum EActivationFunction : int {
        HEAVISIDE_STEP_FUNCTION,
        SIGMOID_FUNCTION,
        HYPERBOLIC_TANGENT_FUNCTION
    };

private:
    EActivationFunction _inputActivationFunction = HEAVISIDE_STEP_FUNCTION;
    EActivationFunction _outputActivationFunction = SIGMOID_FUNCTION;

    int _numInputs{};
    int _numOutputs{};
    int _numHiddenLayers{};
    int _numNeuronsPerHiddenLayer{};
    double _learningRate{};

    std::vector<NeuronLayer> _layers{};

    static double ActivationFunction(const double x, const EActivationFunction activationFunction);

    static double ActivationFunctionDerivative(const double x, const EActivationFunction activationFunction);
    
public:
    NuralNetwork() = default;
    NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, double learningRate);

    void SetInputActivationFunction(const EActivationFunction activationFunction) {
        _inputActivationFunction = activationFunction;
    }

    void SetOutputActivationFunction(const EActivationFunction activationFunction) {
        _outputActivationFunction = activationFunction;
    }

    static double HeavisideStepFunction(const double x) { return x > 0.0 ? 1.0 : 0.0; }

    static double SigmoidFunction(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

    static double HyperbolicTangentFunction(const double x) { return std::tanh(x); }

    static double HeavisideStepFunctionDerivative(const double x) { return 0.0; }

    static double SigmoidFunctionDerivative(const double x) {
        const auto sigmoidOfX = SigmoidFunction(x);
        return sigmoidOfX * (1.0 - sigmoidOfX);
    }

    static double HyperbolicTangentFunctionDerivative(const double x) {
        const auto hyperbolicTangentOfX = HyperbolicTangentFunction(x);
        return 1.0 - hyperbolicTangentOfX * hyperbolicTangentOfX;
    }
};
