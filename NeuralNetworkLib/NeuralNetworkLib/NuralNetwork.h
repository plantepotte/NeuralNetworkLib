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
#include <vector>
#include "NeuronLayer.h"


class NuralNetwork {

private:
    int _numInputs{};
    int _numOutputs{};
    int _numHiddenLayers{};
    int _numNeuronsPerHiddenLayer{};
    double _learningRate{};

    std::vector<NeuronLayer> _layers{};

public:
    EActivationFunction inputActivationFunction{};
    EActivationFunction outputActivationFunction{};
    EActivationFunction hiddenActivationFunction{};
    
    NuralNetwork() = default;
    NuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer, double learningRate);

    void SetInputActivationFunction(const EActivationFunction activationFunction) {
        inputActivationFunction = activationFunction;
    }

    void SetOutputActivationFunction(const EActivationFunction activationFunction) {
        outputActivationFunction = activationFunction;
    }

    Eigen::Vector<double, Eigen::Dynamic> FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs);

    double BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs, const Eigen::Vector<double, Eigen::Dynamic>& targets);

    std::string Train(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& inputs, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& targets, int numEpochs);
};
