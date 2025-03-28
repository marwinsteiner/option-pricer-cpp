#include "models/heston_sv.hpp"
#include <cmath>
#include <random>
#include <complex>

namespace vanilla {
namespace models {

HestonSV::HestonSV(double kappa, double theta, double sigma, double rho, double v0)
    : kappa_(kappa), theta_(theta), sigma_(sigma), rho_(rho), v0_(v0),
      generator_(std::random_device{}()),
      normalDist_(0.0, 1.0) {
}

std::complex<double> HestonSV::characteristicFunction(
    std::complex<double> u, double S, double v0, double T, double r
) const {
    std::complex<double> i(0.0, 1.0);
    
    // Heston model parameters
    double kappa = kappa_;
    double theta = theta_;
    double sigma = sigma_;
    double rho = rho_;
    
    // Improved implementation of the Heston characteristic function
    // Using the formulation from Albrecher et al. (2007) for numerical stability
    
    // First term: deterministic drift
    std::complex<double> term1 = i * u * (std::log(S) + r * T);
    
    // Auxiliary parameters
    std::complex<double> lambda = std::sqrt(
        (kappa - rho * sigma * i * u) * (kappa - rho * sigma * i * u) + 
        sigma * sigma * (i * u + u * u)
    );
    
    std::complex<double> gamma1 = (kappa - rho * sigma * i * u - lambda) / 
                                 (kappa - rho * sigma * i * u + lambda);
    
    std::complex<double> gamma2 = 1.0 - std::exp(-lambda * T);
    std::complex<double> gamma3 = 1.0 - gamma1 * std::exp(-lambda * T);
    
    // Second term: variance contribution
    std::complex<double> term2 = (kappa * theta / (sigma * sigma)) * 
                                ((kappa - rho * sigma * i * u - lambda) * T - 
                                 2.0 * std::log(gamma3 / (1.0 - gamma1)));
    
    // Third term: initial variance contribution
    std::complex<double> term3 = (v0 / (sigma * sigma)) * 
                                ((kappa - rho * sigma * i * u - lambda) * gamma2 / gamma3);
    
    return std::exp(term1 + term2 + term3);
}

StochasticVolModel::PathSimulation HestonSV::simulatePaths(
    double S0, double v0, double T, double r,
    int numPaths, int numSteps
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
            
            // Update variance using square-root (CIR) process
            // Using full truncation scheme for numerical stability
            double vPos = std::max(v, 0.0);
            double dv = kappa_ * (theta_ - vPos) * dt + sigma_ * std::sqrt(vPos) * z2 * sqrtDt;
            v = std::max(0.0, v + dv);  // Ensure variance stays non-negative
            
            // Update spot price (log-normal process)
            double vol = std::sqrt(vPos);
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
