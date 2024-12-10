// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuronLayer.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 15/2/2024
// //Last Modified On : 22/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////


// using Eigen 3.4.0 (https://eigen.tuxfamily.org/index.php?title=Main_Page)
// for linear algebra operations and data structures
#ifndef NEURONLAYER_H
#define NEURONLAYER_H

#include "ActivationLib.h"
#include "../Eigen/Eigen"

class NeuronLayer {
private:
    /**
     * \brief in-place calculation of the outputs of the layer
     */
    void CalcOutputs();

public:
    int numNeurons{}; // Holds the number of neurons in this layer
    int numNeuronInputs{}; // Holds the number of inputs to each neuron
    Eigen::Vector<double, Eigen::Dynamic> outputs{}; // Holds the net outputs of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> inputs{}; // Holds the inputs to each neuron in this layer
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> weights{}; // Holds the weights of each neuron in this layer
    Eigen::Vector<double, Eigen::Dynamic> biases{}; // Holds the biases of each neuron in this layer

    /**
     * \brief construct an empty neuron layer
     */
    NeuronLayer() = default;

    /**
     * \brief construct a neuron layer with a given number of neurons and inputs to each neuron
     * \param numberOfNeurons number of neurons in the layer
     * \param numberOfNeuronInputs number of inputs to each neuron
     */
    NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs);

    /**
     * \brief calculate the outputs of the layer
     * \param inputs vector of inputs to the layer
     * \param activationFunction activation function to apply to the outputs
     * \return 
     */
    Eigen::Vector<double, Eigen::Dynamic> CalcOutputs(const Eigen::Vector<double, Eigen::Dynamic>& inputs,
                                                      EActivationFunction activationFunction);
};
#endif // NEURONLAYER_H
