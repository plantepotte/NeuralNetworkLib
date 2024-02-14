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

enum EActivationFunction : int {
    HEAVISIDE_STEP_FUNCTION,
    SIGMOID_FUNCTION,
    HYPERBOLIC_TANGENT_FUNCTION
};

double ActivationFunction(double x, EActivationFunction activationFunction);

double ActivationFunctionDerivative(double x, EActivationFunction activationFunction);

inline double HeavisideStepFunction(const double x) { return x > 0.0 ? 1.0 : 0.0; }

inline double SigmoidFunction(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline double HyperbolicTangentFunction(const double x) { return std::tanh(x); }

inline double HeavisideStepFunctionDerivative(const double x) { return 0.0; }

inline double SigmoidFunctionDerivative(const double x) {
    const auto sigmoidOfX = SigmoidFunction(x);
    return sigmoidOfX * (1.0 - sigmoidOfX);
}

inline double HyperbolicTangentFunctionDerivative(const double x) {
    const auto hyperbolicTangentOfX = HyperbolicTangentFunction(x);
    return 1.0 - hyperbolicTangentOfX * hyperbolicTangentOfX;
}
