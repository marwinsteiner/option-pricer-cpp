#include "models/stochastic_vol_model.hpp"
#include <cmath>
#include <complex>
#include <functional>
#include <vector>

namespace vanilla {
namespace models {

constexpr double PI = 3.141592653589793238463;

double StochasticVolModel::priceEuropeanCall(
    double S, double K, double v0, double T, double r
) const {
    if (T <= 0.0) return std::max(S - K, 0.0);
    
    double integral = integralForCall(S, K, v0, T, r);
    return integral;
}

double StochasticVolModel::priceEuropeanPut(
    double S, double K, double v0, double T, double r
) const {
    if (T <= 0.0) return std::max(K - S, 0.0);
    
    // Two ways to compute put price:
    // 1. Direct computation using Fourier transform
    // 2. Put-call parity
    
    // Using put-call parity for consistency and accuracy
    double callPrice = priceEuropeanCall(S, K, v0, T, r);
    double discount = std::exp(-r * T);
    double putPrice = callPrice - S + K * discount;
    
    // Ensure non-negative price and no-arbitrage bounds
    putPrice = std::max(0.0, putPrice);
    putPrice = std::min(putPrice, K * discount);  // Put price cannot exceed discounted strike
    putPrice = std::max(putPrice, K * discount - S);  // Put price must satisfy intrinsic value
    
    return putPrice;
}

double StochasticVolModel::integralForCall(
    double S, double K, double v0, double T, double r
) const {
    // Implementation of Fourier transform for call option pricing
    // Using the Carr-Madan approach with a damping factor
    
    const int numPoints = 4096;  // Increased for better accuracy
    const double alpha = 1.5;    // Damping factor
    const double eta = 0.25;     // Grid spacing
    
    std::complex<double> i(0.0, 1.0);
    double discount = std::exp(-r * T);
    double logK = std::log(K);
    double logS = std::log(S);
    
    // Precompute values
    std::vector<double> v(numPoints);
    std::vector<std::complex<double>> ft(numPoints);
    
    // Compute FFT grid points and characteristic function values
    for (int j = 0; j < numPoints; ++j) {
        v[j] = eta * j;
        
        // Damped characteristic function
        std::complex<double> u = v[j] - (alpha + 1.0) * i;
        std::complex<double> cf = characteristicFunction(u, S, v0, T, r);
        
        // Modified characteristic function for call option
        ft[j] = std::exp(-r * T) * cf / (alpha*alpha + alpha - v[j]*v[j] + i*(2*alpha+1)*v[j]);
    }
    
    // Perform numerical integration using trapezoidal rule
    double sum = 0.0;
    for (int j = 0; j < numPoints; ++j) {
        double vj = v[j];
        double weight = (j == 0 || j == numPoints-1) ? 0.5 : 1.0;  // Trapezoidal rule weights
        
        // Real part of the integrand
        double integrand = weight * std::exp(-alpha * logK) * 
                          (ft[j] * std::exp(-i * vj * logK)).real();
        
        sum += integrand;
    }
    
    sum *= eta / PI;
    
    // Apply boundary conditions and no-arbitrage constraints
    double callPrice = sum;
    
    // Ensure non-negative price and no-arbitrage bounds
    callPrice = std::max(0.0, callPrice);
    callPrice = std::min(callPrice, S);  // Call price cannot exceed spot
    callPrice = std::max(callPrice, S - K * discount);  // Call price must satisfy intrinsic value
    
    return callPrice;
}

double StochasticVolModel::integralForPut(
    double S, double K, double v0, double T, double r
) const {
    // Direct computation of put price using Fourier transform
    // This is an alternative to using put-call parity
    
    const int numPoints = 4096;
    const double alpha = 1.5;
    const double eta = 0.25;
    
    std::complex<double> i(0.0, 1.0);
    double discount = std::exp(-r * T);
    double logK = std::log(K);
    
    // Precompute values
    std::vector<double> v(numPoints);
    std::vector<std::complex<double>> ft(numPoints);
    
    // Compute FFT grid points and characteristic function values
    for (int j = 0; j < numPoints; ++j) {
        v[j] = eta * j;
        
        // Damped characteristic function for put option
        std::complex<double> u = v[j] - (alpha + 1.0) * i;
        std::complex<double> cf = characteristicFunction(-u, S, v0, T, r);  // Note the negative u
        
        // Modified characteristic function for put option
        ft[j] = std::exp(-r * T) * cf / (alpha*alpha + alpha - v[j]*v[j] + i*(2*alpha+1)*v[j]);
    }
    
    // Perform numerical integration
    double sum = 0.0;
    for (int j = 0; j < numPoints; ++j) {
        double vj = v[j];
        double weight = (j == 0 || j == numPoints-1) ? 0.5 : 1.0;
        
        double integrand = weight * std::exp(-alpha * logK) * 
                          (ft[j] * std::exp(-i * vj * logK)).real();
        
        sum += integrand;
    }
    
    sum *= eta / PI;
    
    // Apply put-call parity to ensure consistency
    double putPrice = sum + K * discount - S;
    
    // Ensure non-negative price and no-arbitrage bounds
    putPrice = std::max(0.0, putPrice);
    putPrice = std::min(putPrice, K * discount);  // Put price cannot exceed discounted strike
    putPrice = std::max(putPrice, K * discount - S);  // Put price must satisfy intrinsic value
    
    return putPrice;
}

} // namespace models
} // namespace vanilla
