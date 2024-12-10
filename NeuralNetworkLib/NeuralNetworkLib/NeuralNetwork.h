// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuralNetwork.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 22/2/2024
// //Last Modified On : 22/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#pragma once
#include <vector>
#include "NeuronLayer.h"


class NeuralNetwork {
private:
    int _numInputs{};
    int _numOutputs{};
    int _numHiddenLayers{};
    int _numNeuronsPerHiddenLayer{};
    double _learningRate{};

    std::vector<NeuronLayer> _layers{};
    std::vector<Eigen::Vector<double, Eigen::Dynamic>> _neuronDeltas{};

    /**
     * \brief update the weights and biases of a layer
     * \param grad gradient vector to use
     * \param i layer index
     */
    void UpdateWeightsAndBiases(const Eigen::Vector<double, Eigen::Dynamic>& grad, int i);

    EActivationFunction _outputActivationFunction{};
    EActivationFunction _hiddenActivationFunction{};

public:
    /**
     * \brief Construct empty neural network
     */
    NeuralNetwork() = default;

    /**
     * \brief Construct a neural network with a given parameters
     * \param numInputs number of inputs to the network
     * \param numOutputs number of outputs from the network
     * \param numHiddenLayers number of hidden layers in the network
     * \param numNeuronsPerHiddenLayer number of neurons in each hidden layer
     * \param learningRate learning rate of the network
     */
    NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer,
                  double learningRate);

    /**
     * \brief set the activation function of the output layer
     * \param activationFunction given activation function
     */
    void SetOutputActivationFunction(const EActivationFunction activationFunction) {
        _outputActivationFunction = activationFunction;
    }

    /**
     * \brief set the activation function of the hidden layers
     * \param activationFunction given activation function
     */
    void SetHiddenActivationFunction(const EActivationFunction activationFunction) {
        _hiddenActivationFunction = activationFunction;
    }

    /**
     * \brief feed forward the inputs through the network
     * \param inputs input vector
     * \return output vector
     */
    Eigen::Vector<double, Eigen::Dynamic> FeedForward(const Eigen::Vector<double, Eigen::Dynamic>& inputs);

    /**
     * \brief back propagate the error through the network
     * \param inputs vector of inputs
     * \param targets vector of targets
     * \return 
     */
    double BackPropagate(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
                         const Eigen::Vector<double, Eigen::Dynamic>& targets);

    /**
     * \brief train the network for a given number of epochs
     * \param inputs vector of input vectors
     * \param targets vector of target vectors
     * \param numEpochs number of epochs to train
     * \return 
     */
    std::string Train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets,
                      int numEpochs);

    /**
     * \brief train the network until the error is below a given threshold
     *  or the number of epochs exceeds a given maximum
     * \param inputs vector of input vectors
     * \param targets vector of target vectors
     * \param maxError error threshold
     * \param maxEpochs maximum number of epochs
     * \return 
     */
    std::string Train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets,
                      double maxError = 1e-3, int maxEpochs = 1000);

    bool SaveToFile(const std::string& filename);

    bool LoadFromFile(const std::string& filename);
     
};
