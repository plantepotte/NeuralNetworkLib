// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: NeuralNetworkLib.cpp
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 15/2/2024
// //Last Modified On : 15/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#include <iostream>

#include "NeuralNetworkLib/NuralNetwork.h"

int main(int argc, char* argv[]) {
    NuralNetwork nuralNetwork(2, 1, 3, 2, 0.098);
    nuralNetwork.SetInputActivationFunction(EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION);
    nuralNetwork.SetOutputActivationFunction(EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION);
    nuralNetwork.SetHiddenActivationFunction(EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION);
    
    std::cout << nuralNetwork.Train(std::vector<std::vector<double>>{
                                        std::vector<double>{0, 0},
                                        std::vector<double>{0, 1},
                                        std::vector<double>{1, 0},
                                        std::vector<double>{1, 1}
                                    }
                                    , std::vector<std::vector<double>>{
                                        std::vector<double>{0},
                                        std::vector<double>{1},
                                        std::vector<double>{1},
                                        std::vector<double>{0}
                                    }, 1e-2, 1e5) <<
        '\n';

    std::cout << std::fixed;
    std::cout << "0, 0: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({0, 0}))[0] << '\n';
    std::cout << "0, 1: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({0, 1}))[0] << '\n';
    std::cout << "1, 0: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({1, 0}))[0] << '\n';
    std::cout << "1, 1: " << nuralNetwork.FeedForward(Eigen::Vector<double, 2>({1, 1}))[0] << '\n';

    return 0;
}
