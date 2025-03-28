#pragma once

#include "models/stochastic_vol_model.hpp"
#include <random>

namespace vanilla {
    namespace models {

        class LogNormalSV : public StochasticVolModel {
        public:
            LogNormalSV(
                double kappa,    // Mean reversion speed
                double theta,    // Long-term volatility
                double sigma,    // Volatility of volatility
                double rho       // Correlation between spot and vol
            );

            // Implement the characteristic function
            std::complex<double> characteristicFunction(
                std::complex<double> u, double S, double v0, double T, double r
            ) const override;

            // Implement Monte Carlo simulation
            PathSimulation simulatePaths(
                double S0, double v0, double T, double r,
                int numPaths, int numSteps
            ) const override;

            // Getters for model parameters
            double getKappa() const { return kappa_; }
            double getTheta() const { return theta_; }
            double getSigma() const { return sigma_; }
            double getRho() const { return rho_; }

        private:
            double kappa_;   // Mean reversion speed
            double theta_;   // Long-term volatility
            double sigma_;   // Volatility of volatility
            double rho_;     // Correlation between spot and vol

            // Random number generation
            mutable std::mt19937 generator_;
            mutable std::normal_distribution<double> normalDist_;
        };

    } // namespace models
} // namespace vanilla
