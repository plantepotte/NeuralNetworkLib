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
#include <vector>

class Neuron;

class NeuronLayer {
private:

public:
    int numNeurons{}; // Holds the number of neurons in this layer
    int numNeuronInputs{}; // Holds the number of inputs to each neuron

    std::vector<Neuron> neurons{};

    NeuronLayer() = default;
    NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs);
    
};
