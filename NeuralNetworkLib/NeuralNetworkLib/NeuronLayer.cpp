#include "NeuronLayer.h"

#include "Neuron.h"

NeuronLayer::NeuronLayer(int numberOfNeurons, int numberOfNeuronInputs): numNeurons(numberOfNeurons), numNeuronInputs(numberOfNeuronInputs) {
    neurons = std::vector<Neuron>(numberOfNeurons, Neuron());
}
