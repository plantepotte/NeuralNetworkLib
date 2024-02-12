// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: Neuron.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 12/2/2024
// //Last Modified On : 12/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#pragma once
#include <vector>
#include <string>

/**
 * \brief Struct to hold a training set for the neuron
 */
struct TrainingSet {
    std::vector<double> input{};
    double output{};
    TrainingSet() = default;

    /**
     * \brief Create a training set
     * \param Input Vector of input values
     * \param Output Expected output
     */
    TrainingSet(const std::vector<double>& Input, double Output) : input(Input), output(Output) {
    }
};

class Neuron {
private:
    double _output{};
    double _errorGradient{};
    double _netOutput{};
    double _bias{};

    std::vector<double> _weights{};
    std::vector<double> _inputs{};

    /**
     * \brief Perform a biased dot product
     * \param weights Vector of weights
     * \param inputs Vector of inputs
     * \param bias bias to add to the dot product
     * \return value of the biased dot product
     */
    static double BiasedDotProd(const std::vector<double>& weights, const std::vector<double>& inputs, double bias);

    /**
     * \brief Heaviside step function
     * \param x input value
     * \return 1 if x > 0, 0 otherwise
     */
    static double StepFunction(const double x) { return x > 0 ? 1 : 0; }

    /**
     * \brief train the neuron for one epoch
     * \param j epoch number
     * \return information about the training result
     */
    std::string TrainEpoch(int j);

public:
    /**
     * \brief construct uninitialized neuron
     */
    Neuron() = default;

    Neuron(const std::vector<double>& weights, const std::vector<double>& inputs) : _weights(weights), _inputs(inputs) {
    }

    /**
     * \brief train the neuron for a number of epochs
     * \param epochs number of epochs to train for
     * \return information about the training result
     */
    std::string Train(int epochs = 1);

    /**
     * \brief train the neuron until the root mean square error is below a tolerance,
     * or a limit for the number of epochs is reached
     * \param tolerance tolerance for the root mean square error
     * \param epochLimit limit for the number of epochs to train for
     * \return information about the training result
     */
    std::string Train(double tolerance, int epochLimit = 100);

    /**
     * \brief calculate the output of the neuron for given inputs
     * \param i1 first input
     * \param i2 second input
     * \return result of the neuron
     */
    double CalcOutput(double i1, double i2) const { return StepFunction(BiasedDotProd(_weights, {i1, i2}, _bias)); }

    /**
     * \brief get copy of the weights currently used by the neuron
     * \return vector of weights
     */
    std::vector<double> Weights() const { return _weights; }

    /**
     * \brief get weights currently used by the neuron
     * \param weights vector of weights to use
     */
    void SetWeights(const std::vector<double>& weights) { _weights = weights; }

    /**
     * \brief get the bias currently used by the neuron
     * \return value of the bias
     */
    double Bias() const { return _bias; }

    /**
     * \brief set the bias to be used by the neuron
     * \param bias bias to use
     */
    void SetBias(double bias) { _bias = bias; }

    /**
     * \brief get the current error gradient
     * \return error gradient
     */
    double ErrorGradient() const { return _errorGradient; }

    /**
     * \brief set the error gradient to be used by the neuron
     * \param errorGradient error gradient to use
     */
    void SetErrorGradient(double errorGradient) { _errorGradient = errorGradient; }

    /**
     * \brief get the current inputs used by the neuron
     * \return vector of inputs
     */
    std::vector<double> Inputs() const { return _inputs; }

    /**
     * \brief set the inputs to be used by the neuron
     * \param inputs vector of inputs to use
     */
    void SetInputs(const std::vector<double>& inputs) { _inputs = inputs; }
};
