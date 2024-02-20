#include "ActivationLib.h"

#include <stdexcept>

double ActivationLib::ActivationFunction(double x, EActivationFunction activationFunction) {
    switch (activationFunction) {
    case EActivationFunction::HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunction(x);
    case EActivationFunction::SIGMOID_FUNCTION:
        return SigmoidFunction(x);
    case EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunction(x);
    case EActivationFunction::RELU_FUNCTION:
        return ReLUFunction(x);
    case EActivationFunction::NONE:
        return x;
    }
    throw std::invalid_argument("Invalid activation function");
}

double ActivationLib::ActivationFunctionDerivative(double x, EActivationFunction activationFunction) {
    switch (activationFunction) {
    case EActivationFunction::HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunctionDerivative();
    case EActivationFunction::SIGMOID_FUNCTION:
        return SigmoidFunctionDerivative(x);
    case EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunctionDerivative(x);
    case EActivationFunction::RELU_FUNCTION:
        return ReLUFunctionDerivative(x);
    case EActivationFunction::NONE:
        return 1.;
    }
    throw std::invalid_argument("Invalid activation function");
}