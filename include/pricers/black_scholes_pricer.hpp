#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "models/black_scholes.hpp"

namespace vanilla {
    namespace pricers {

        enum class OptionType {
            CALL = 'C',
            PUT = 'P',
            INVERSE_CALL,  // Changed from 'IC'
            INVERSE_PUT    // Changed from 'IP'
        };

        struct OptionData {
            double spot;
            double strike;
            double timeToMaturity;
            double rateAnnual;
            double volatilityAnnual;
            OptionType optionType;  // Changed from char to OptionType
        };

        struct Greeks {
            double delta;
            double gamma;
            double vega;
            double theta;
            double rho;
        };

        class BlackScholesPricer {
        public:
            BlackScholesPricer() = default;

            // Single option pricing
            double price(const OptionData& option) const;
            Greeks computeGreeks(const OptionData& option) const;

            // Batch pricing
            Eigen::VectorXd priceVector(const std::vector<OptionData>& options) const;
            std::vector<Greeks> computeGreeksVector(const std::vector<OptionData>& options) const;

            // Implied volatility
            double impliedVolatility(
                const OptionData& option,
                double targetPrice,
                double initialGuess = 0.3,
                double tolerance = 1e-8,
                int maxIterations = 100
            ) const;

            // Batch implied volatility
            Eigen::VectorXd impliedVolatilityVector(
                const std::vector<OptionData>& options,
                const Eigen::VectorXd& targetPrices,
                double initialGuess = 0.3,
                double tolerance = 1e-8,
                int maxIterations = 100
            ) const;

        private:
            // Helper methods
            double priceCall(const OptionData& option) const;
            double pricePut(const OptionData& option) const;
            double priceInverseCall(const OptionData& option) const;
            double priceInversePut(const OptionData& option) const;

            // Newton-Raphson for implied vol
            double newtonRaphsonVol(
                const OptionData& option,
                double targetPrice,
                double initialGuess,
                double tolerance,
                int maxIterations
            ) const;

            // Vega calculation for implied vol
            double computeVega(const OptionData& option) const;
        };

    } // namespace pricers
} // namespace vanilla