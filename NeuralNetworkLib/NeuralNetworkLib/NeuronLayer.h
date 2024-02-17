// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuronLayer.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 12/2/2024
// //Last Modified On : 12/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#pragma once
// using Eigen 3.4.0 (https://eigen.tuxfamily.org/index.php?title=Main_Page)
// for linear algebra operations and data structures
#include "ActivationLib.h"
#include "Eigen/Eigen"

class Neuron;

class NeuronLayer {
private:

    void CalcOutputs();

public:
    int numNeurons{}; // Holds the number of neurons in this layer
    int numNeuronInputs{}; // Holds the number of inputs to each neuron
    EActivationFunction activationFunction{}; // Which activation function to use
    Eigen::Vector<double, Eigen::Dynamic> outputs{}; // Holds the outputs of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> inputs{}; // Holds the inputs to each neuron in this layer
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights{}; // Holds the weights of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> biases{}; // Holds the biases of each neuron in this layer


    NeuronLayer() = default;
    NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs);
    

    Eigen::Vector<double, Eigen::Dynamic> CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& inputs);
    
    
};
