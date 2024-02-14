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
#include "ActivationLib.h"
#include <vector>

#include "Eigen/Eigen"

#include "NeuronLayer.h"


class NuralNetwork {

private:
    EActivationFunction _inputActivationFunction = HEAVISIDE_STEP_FUNCTION;
    EActivationFunction _outputActivationFunction = SIGMOID_FUNCTION;

    int _numInputs{};
    int _numOutputs{};
    int _numHiddenLayers{};
    int _numNeuronsPerHiddenLayer{};
    double _learningRate{};

    std::vector<NeuronLayer> _layers{};

public:
    NuralNetwork() = default;
    NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, double learningRate);

    void SetInputActivationFunction(const EActivationFunction activationFunction) {
        _inputActivationFunction = activationFunction;
    }

    void SetOutputActivationFunction(const EActivationFunction activationFunction) {
        _outputActivationFunction = activationFunction;
    }

    Eigen::Vector<double, Eigen::Dynamic> FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs);

    void BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs, const Eigen::Vector<double, Eigen::Dynamic>& targets);
};
