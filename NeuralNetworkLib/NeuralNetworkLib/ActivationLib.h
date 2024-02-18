// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: ActivationLib.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 14/2/2024
// //Last Modified On : 14/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////

#pragma once
#include <cmath>

enum class EActivationFunction {
    HEAVISIDE_STEP_FUNCTION,
    SIGMOID_FUNCTION,
    HYPERBOLIC_TANGENT_FUNCTION,
    RELU_FUNCTION
};

class ActivationLib {
public:

    static constexpr double pi = 3.14159265358979323846;

    // f(x) : R -> R
    static double ActivationFunction(double x, EActivationFunction activationFunction);

    static double ActivationFunctionDerivative(double x, EActivationFunction activationFunction);

    static double HeavisideStepFunction(const double x) { return x > 0.0 ? 1.0 : 0.0; }

    static double SigmoidFunction(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

    static double HyperbolicTangentFunction(const double x) { return std::tanh(x); }

    static double ReLUFunction(const double x) { return x > 0.0 ? x : 0.0; }

    // f'(x) : R -> R
    static double HeavisideStepFunctionDerivative() { return 0.0; }

    static double SigmoidFunctionDerivative(const double x) {
        const auto sigmoidOfX = SigmoidFunction(x);
        return sigmoidOfX * (1.0 - sigmoidOfX);
    }

    static double HyperbolicTangentFunctionDerivative(const double x) {
        const auto hyperbolicTangentOfX = HyperbolicTangentFunction(x);
        return 1.0 - hyperbolicTangentOfX * hyperbolicTangentOfX;
    }

    static double ReLUFunctionDerivative(const double x) { return x > 0.0 ? 1.0 : 0.0; }
};
