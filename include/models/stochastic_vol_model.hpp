#pragma once

#include <complex>
#include <functional>
#include <Eigen/Dense>

namespace vanilla {
    namespace models {

        class StochasticVolModel {
        public:
            virtual ~StochasticVolModel() = default;

            // Characteristic function (moment generating function)
            virtual std::complex<double> characteristicFunction(
                std::complex<double> u, double S, double v0, double T, double r
            ) const = 0;

            // Fourier transform pricing
            double priceEuropeanCall(
                double S, double K, double v0, double T, double r
            ) const;

            double priceEuropeanPut(
                double S, double K, double v0, double T, double r
            ) const;

            // Monte Carlo simulation
            struct PathSimulation {
                Eigen::VectorXd timePoints;
                Eigen::MatrixXd spotPaths;
                Eigen::MatrixXd volPaths;
            };

            virtual PathSimulation simulatePaths(
                double S0, double v0, double T, double r,
                int numPaths, int numSteps
            ) const = 0;

            // Helpers for Fourier pricing
            protected:
                double integralForCall(double S, double K, double v0, double T, double r) const;
                double integralForPut(double S, double K, double v0, double T, double r) const;
        };

    } // namespace models
} // namespace vanilla
