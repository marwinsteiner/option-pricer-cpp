#include "models/log_normal_sv.hpp"
#include <cmath>
#include <random>
#include <complex>
#include <Eigen/Dense>

namespace vanilla {
namespace models {

LogNormalSV::LogNormalSV(double kappa, double theta, double sigma, double rho)
    : kappa_(kappa), theta_(theta), sigma_(sigma), rho_(rho) {}

std::complex<double> LogNormalSV::characteristicFunction(
    std::complex<double> u, double S, double v0, double T, double r
) const {
    // Implementation of the characteristic function for the Log-Normal SV model
    // Following Sepp's approach
    std::complex<double> i(0.0, 1.0);
    
    // Adjust for the risk-neutral measure
    double forward = std::log(S) + r * T;
    
    // Parameters for the characteristic function
    std::complex<double> lambda = std::sqrt(
        kappa_ * kappa_ - 2.0 * sigma_ * sigma_ * i * u * (i * u - 1.0)
    );
    
    std::complex<double> d = std::sqrt(
        (kappa_ - 2.0 * rho_ * sigma_ * i * u) * (kappa_ - 2.0 * rho_ * sigma_ * i * u) - 
        sigma_ * sigma_ * (i * u * i * u - i * u)
    );
    
    std::complex<double> g1 = (kappa_ - 2.0 * rho_ * sigma_ * i * u - d) / 
                             (kappa_ - 2.0 * rho_ * sigma_ * i * u + d);
    
    // Time-dependent terms
    std::complex<double> C = kappa_ * theta_ / (sigma_ * sigma_) * 
                           (
                               (kappa_ - 2.0 * rho_ * sigma_ * i * u - d) * T - 
                               2.0 * std::log((1.0 - g1 * std::exp(-d * T)) / (1.0 - g1))
                           );
    
    std::complex<double> D = (kappa_ - 2.0 * rho_ * sigma_ * i * u - d) / 
                           (sigma_ * sigma_) * 
                           (
                               (1.0 - std::exp(-d * T)) / 
                               (1.0 - g1 * std::exp(-d * T))
                           );
    
    // Combine terms for the final characteristic function
    return std::exp(i * u * forward + C + D * v0);
}

StochasticVolModel::PathSimulation LogNormalSV::simulatePaths(
    double S0, double v0, double T, double r, int numPaths, int numSteps
) const {
    // Initialize result structure
    PathSimulation result;
    
    // Initialize time points
    result.timePoints = Eigen::VectorXd::LinSpaced(numSteps + 1, 0.0, T);
    
    // Initialize paths matrices
    result.spotPaths = Eigen::MatrixXd::Zero(numSteps + 1, numPaths);
    result.volPaths = Eigen::MatrixXd::Zero(numSteps + 1, numPaths);
    
    // Set initial values
    result.spotPaths.row(0).setConstant(S0);
    result.volPaths.row(0).setConstant(std::sqrt(v0));  // Store volatility, not variance
    
    // Time step
    double dt = T / numSteps;
    double sqrtDt = std::sqrt(dt);
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    
    // Simulate paths
    #pragma omp parallel for
    for (int i = 0; i < numPaths; ++i) {
        double S = S0;
        double v = v0;  // This is variance (volatility squared)
        
        for (int j = 0; j < numSteps; ++j) {
            // Generate correlated random numbers
            double z1 = normal(gen);
            double z2 = rho_ * z1 + std::sqrt(1.0 - rho_ * rho_) * normal(gen);
            
            // Update volatility (using log-normal process)
            double vol = std::sqrt(v);  // Current volatility
            double dv = kappa_ * (theta_ - v) * dt + sigma_ * vol * z2 * sqrtDt;
            v = std::max(1e-8, v + dv);  // Ensure variance stays positive
            
            // Update spot price (using log-normal process)
            double dS = r * S * dt + vol * S * z1 * sqrtDt;
            S = S + dS;
            
            // Store updated values
            result.spotPaths(j + 1, i) = S;
            result.volPaths(j + 1, i) = std::sqrt(v);  // Store volatility, not variance
        }
    }
    
    return result;
}

} // namespace models
} // namespace vanilla
