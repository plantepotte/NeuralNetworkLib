// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
// //FileName: ActivationLib.h
// //FileType: Visual C++ Source file
// //Author : Anders P. Åsbø
// //Created On : 15/2/2024
// //Last Modified On : 22/2/2024
// //Description :
// //////////////////////////////////////////////////////////////////////////
// //////////////////////////////
#ifndef ACTIVATIONLIB_H
#define ACTIVATIONLIB_H

#include <cmath>

/**
 * \brief Enum class to represent the different activation functions
 */
enum class EActivationFunction {
    HEAVISIDE_STEP_FUNCTION,
    SIGMOID_FUNCTION,
    HYPERBOLIC_TANGENT_FUNCTION,
    RELU_FUNCTION,
    NONE // no activation function - linear activation
};

class ActivationLib {
public:
    // constants if needed
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double e = 2.71828182845904523536;
 
    /**
     * \brief calculate the activation function of a given input
     * \param x input to the activation function
     * \param activationFunction indicates which activation function to use
     * \return activated output
     */
    static double ActivationFunction(double x, EActivationFunction activationFunction);

    /**
     * \brief calculate the derivative of the activation function of a given input
     * \param x input to the activation function
     * \param activationFunction which activation function to use
     * \return value of the derivative of the activation function
     */
    static double ActivationFunctionDerivative(double x, EActivationFunction activationFunction);

    // f(x) : R -> R
    /**
     * \brief Heaviside step function
     * \param x input
     * \return output
     */
    static double HeavisideStepFunction(const double x) { return x > 0.0 ? 1.0 : 0.0; }

    /**
     * \brief Sigmoid/logistic function
     * \param x input
     * \return output
     */
    static double SigmoidFunction(const double x) { return 1.0 / (1.0 + std::exp(-x)); }

    /**
     * \brief tanh hyperbolic tangent function
     * \param x input
     * \return output
     */
    static double HyperbolicTangentFunction(const double x) { return std::tanh(x); }

    /**
     * \brief Rectified Linear Unit (ReLU) function
     * \param x input
     * \return output
     */
    static double ReLUFunction(const double x) { return x > 0.0 ? x : 0.0; }

    // function to calculate the derivative of the activation functions
    // f'(x) : R -> R
    /**
     * \brief derivative of the Heaviside step function
     * \return output
     */
    static double HeavisideStepFunctionDerivative() { return 1.0; }

    /**
     * \brief derivative of the sigmoid/logistic function
     * \param x input
     * \return output
     */
    static double SigmoidFunctionDerivative(const double x) {
        const auto sigmoidOfX = SigmoidFunction(x);
        return sigmoidOfX * (1.0 - sigmoidOfX);
    }

    /**
     * \brief derivative of the tanh hyperbolic tangent function
     * \param x input
     * \return output
     */
    static double HyperbolicTangentFunctionDerivative(const double x) {
        const auto hyperbolicTangentOfX = HyperbolicTangentFunction(x);
        return 1.0 - hyperbolicTangentOfX * hyperbolicTangentOfX;
    }

    /**
     * \brief derivative of the Rectified Linear Unit (ReLU) function
     * \param x input
     * \return output
     */
    static double ReLUFunctionDerivative(const double x) { return x > 0.0 ? 1.0 : 0.0; }
};
#endif // ACTIVATIONLIB_H
