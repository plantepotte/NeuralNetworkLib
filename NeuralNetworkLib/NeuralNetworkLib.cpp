
#include <iostream>

#include "NeuralNetworkLib/NuralNetwork.h"

int main(int argc, char* argv[])
{
    NuralNetwork nuralNetwork(2, 1, 1, 2, 0.1);
    nuralNetwork.SetInputActivationFunction(EActivationFunction::SIGMOID_FUNCTION);
    nuralNetwork.SetOutputActivationFunction(EActivationFunction::HEAVISIDE_STEP_FUNCTION);
    std::cout << nuralNetwork.Train(Eigen::Matrix<double, 2, 4>({ {0, 0, 1, 1}, {0, 1, 0, 1} }), Eigen::Matrix<double, 1, 4>({ {0, 1, 1, 1} }), 1000) <<
        '\n';

    std::cout << "0, 0: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({ 0, 0 }))[0] << '\n';
    std::cout << "0, 1: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({ 0, 1 }))[0] << '\n';
    std::cout << "1, 0: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({ 1, 0 }))[0] << '\n';
    std::cout << "1, 1: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({ 1, 1 }))[0] << '\n';
    return 0;
}
