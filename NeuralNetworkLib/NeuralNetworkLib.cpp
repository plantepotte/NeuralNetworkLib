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

#include "NeuralNetworkLib/NeuralNetwork.h"

int main(int argc, char* argv[]) {
    
    NeuralNetwork neuralNetwork(3, 3, 2, 9, 0.2);
    neuralNetwork.SetHiddenActivationFunction(EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION);
    neuralNetwork.SetOutputActivationFunction(EActivationFunction::SIGMOID_FUNCTION);

    const auto inputs = std::vector<std::vector<double>>{
        std::vector<double>{0, 0, 0},
        std::vector<double>{0, 0, 1},
        std::vector<double>{0, 1, 0},
        std::vector<double>{0, 1, 1},
        std::vector<double>{1, 0, 0},
        std::vector<double>{1, 0, 1},
        std::vector<double>{1, 1, 0},
        std::vector<double>{1, 1, 1}
    };

    const auto targets = std::vector<std::vector<double>>{
        std::vector<double>{0, 0, 1},
        std::vector<double>{0, 1, 0},
        std::vector<double>{0, 1, 1},
        std::vector<double>{1, 0, 0},
        std::vector<double>{1, 0, 1},
        std::vector<double>{1, 1, 0},
        std::vector<double>{1, 1, 1},
        std::vector<double>{0, 0, 0}
    };
    
    std::cout << neuralNetwork.Train(inputs, targets, 1e-1, static_cast<int>(1e6)) << '\n';
    
    // std::cout << std::fixed;
    for (auto& input : inputs) {
        std::cout << "Input: " << input[0] << " " << input[1] << " " << input[2] << '\n';
        std::cout << "Output: " << Eigen::rint(
            neuralNetwork.FeedForward(Eigen::Vector3d{input[0], input[1], input[2]}).transpose().array()) << "\n\n";
    }

    return 0;
}
