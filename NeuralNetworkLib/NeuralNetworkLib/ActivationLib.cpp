#include "ActivationLib.h"

#include <stdexcept>

double ActivationLib::ActivationFunction(const double x, EActivationFunction activationFunction) {
    switch (activationFunction) {
    case EActivationFunction::HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunction(x);
    case EActivationFunction::SIGMOID_FUNCTION:
        return SigmoidFunction(x);
    case EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunction(x);
    }
    throw std::invalid_argument("Invalid activation function");
}

double ActivationLib::ActivationFunctionDerivative(const double x, EActivationFunction activationFunction) {
    switch (activationFunction) {
    case EActivationFunction::HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunctionDerivative();
    case EActivationFunction::SIGMOID_FUNCTION:
        return SigmoidFunctionDerivative(x);
    case EActivationFunction::HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunctionDerivative(x);
    }
    throw std::invalid_argument("Invalid activation function");
}