#include "pricers/normal_pricer.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace vanilla {
    namespace pricers {

        double NormalPricer::price(const OptionData& option) const {
            switch (option.optionType) {
                case OptionType::CALL: return priceCall(option);
                case OptionType::PUT: return pricePut(option);
                case OptionType::INVERSE_CALL: return priceInverseCall(option);
                case OptionType::INVERSE_PUT: return priceInversePut(option);
                default:
                    throw std::invalid_argument("Invalid option type");
            }
        }

        double NormalPricer::priceCall(const OptionData& option) const {
            return models::Normal::price_call(
                option.spot,
                option.strike,
                option.timeToMaturity,
                option.rateAnnual,
                option.volatilityAnnual
            );
        }

        double NormalPricer::pricePut(const OptionData& option) const {
            return models::Normal::price_put(
                option.spot,
                option.strike,
                option.timeToMaturity,
                option.rateAnnual,
                option.volatilityAnnual
            );
        }

        double NormalPricer::priceInverseCall(const OptionData& option) const {
            // For inverse options, we swap spot and strike
            OptionData inversedOption = option;
            std::swap(inversedOption.spot, inversedOption.strike);
            return priceCall(inversedOption);
        }

        double NormalPricer::priceInversePut(const OptionData& option) const {
            OptionData inversedOption = option;
            std::swap(inversedOption.spot, inversedOption.strike);
            return pricePut(inversedOption);
        }

        Greeks NormalPricer::computeGreeks(const OptionData& option) const {
            Greeks greeks;

            if (option.optionType == OptionType::CALL) {
                greeks.delta = models::Normal::delta_call(
                    option.spot, option.strike, option.timeToMaturity,
                    option.rateAnnual, option.volatilityAnnual
                );
                greeks.theta = models::Normal::theta_call(
                    option.spot, option.strike, option.timeToMaturity,
                    option.rateAnnual, option.volatilityAnnual
                );
            } else if (option.optionType == OptionType::PUT) {
                greeks.delta = models::Normal::delta_put(
                    option.spot, option.strike, option.timeToMaturity,
                    option.rateAnnual, option.volatilityAnnual
                );
                greeks.theta = models::Normal::theta_put(
                    option.spot, option.strike, option.timeToMaturity,
                    option.rateAnnual, option.volatilityAnnual
                );
            } else {
                throw std::invalid_argument("Invalid option type for Greeks calculation");
            }

            greeks.gamma = models::Normal::gamma(
                option.spot, option.strike, option.timeToMaturity,
                option.rateAnnual, option.volatilityAnnual
            );

            greeks.vega = models::Normal::vega(
                option.spot, option.strike, option.timeToMaturity,
                option.rateAnnual, option.volatilityAnnual
            );

            return greeks;
        }

        Eigen::VectorXd NormalPricer::priceVector(
            const std::vector<OptionData>& options
        ) const {
            const size_t n = options.size();
            Eigen::VectorXd prices(n);

            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                prices(i) = price(options[i]);
            }

            return prices;
        }

        std::vector<Greeks> NormalPricer::computeGreeksVector(
            const std::vector<OptionData>& options
        ) const {
            const size_t n = options.size();
            std::vector<Greeks> greeks(n);

            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                greeks[i] = computeGreeks(options[i]);
            }

            return greeks;
        }

        double NormalPricer::computeVega(const OptionData& option) const {
            return models::Normal::vega(
                option.spot,
                option.strike,
                option.timeToMaturity,
                option.rateAnnual,
                option.volatilityAnnual
            );
        }

        double NormalPricer::impliedVolatility(
            const OptionData& option,
            double targetPrice,
            double initialGuess,
            double tolerance,
            int maxIterations
        ) const {
            return newtonRaphsonVol(option, targetPrice, initialGuess, tolerance, maxIterations);
        }

        double NormalPricer::newtonRaphsonVol(
            const OptionData& option,
            double targetPrice,
            double initialGuess,
            double tolerance,
            int maxIterations
        ) const {
            double sigma = initialGuess;
            OptionData iterOption = option;

            for (int i = 0; i < maxIterations; ++i) {
                iterOption.volatilityAnnual = sigma;
                double price = this->price(iterOption);
                double vega = this->computeVega(iterOption);

                if (std::abs(vega) < 1e-10) {
                    throw std::runtime_error("Vega too close to zero");
                }

                double diff = price - targetPrice;
                if (std::abs(diff) < tolerance) {
                    return sigma;
                }

                sigma -= diff / vega;

                // Bounds check
                if (sigma <= 0.0001 || sigma > 5.0) {
                    throw std::runtime_error("Implied volatility out of bounds");
                }
            }

            throw std::runtime_error("Implied volatility did not converge");
        }

        Eigen::VectorXd NormalPricer::impliedVolatilityVector(
            const std::vector<OptionData>& options,
            const Eigen::VectorXd& targetPrices,
            double initialGuess,
            double tolerance,
            int maxIterations
        ) const {
            const size_t n = options.size();
            if (n != static_cast<size_t>(targetPrices.size())) {
                throw std::invalid_argument("Size mismatch between options and prices");
            }

            Eigen::VectorXd impliedVols(n);

            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                impliedVols(i) = impliedVolatility(
                    options[i], targetPrices(i), initialGuess, tolerance, maxIterations
                );
            }

            return impliedVols;
        }

    } // namespace pricers
} // namespace vanilla
