#include "ActivationLib.h"

#include <stdexcept>

double ActivationFunction(const double x, const EActivationFunction activationFunction) {
    switch (activationFunction) {
    case HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunction(x);
    case SIGMOID_FUNCTION:
        return SigmoidFunction(x);
    case HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunction(x);
    }
    throw std::invalid_argument("Invalid activation function");
}

double ActivationFunctionDerivative(const double x, const EActivationFunction activationFunction) {
    switch (activationFunction) {
    case HEAVISIDE_STEP_FUNCTION:
        return HeavisideStepFunctionDerivative(x);
    case SIGMOID_FUNCTION:
        return SigmoidFunctionDerivative(x);
    case HYPERBOLIC_TANGENT_FUNCTION:
        return HyperbolicTangentFunctionDerivative(x);
    }
    throw std::invalid_argument("Invalid activation function");
}