﻿// //////////////////////////////////////////////////////////////////////////
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
#include "Eigen/Eigen"

enum EActivationFunction : int;
class Neuron;

class NeuronLayer {
private:
    Eigen::Vector<double, Eigen::Dynamic> _outputs{}; // Holds the outputs of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> _inputs{}; // Holds the inputs to each neuron in this layer
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> _weights{}; // Holds the weights of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> _biases{}; // Holds the biases of each neuron in this layer

    void CalcOutputs();

public:
    int numNeurons{}; // Holds the number of neurons in this layer
    int numNeuronInputs{}; // Holds the number of inputs to each neuron
    EActivationFunction activationFunction{}; // Which activation function to use


    NeuronLayer() = default;
    NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs);

    Eigen::Vector<double, Eigen::Dynamic> GetOutputs() const { return _outputs; }

    Eigen::Vector<double, Eigen::Dynamic> CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& inputs);
};
