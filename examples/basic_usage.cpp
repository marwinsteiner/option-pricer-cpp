#include "pricers/black_scholes_pricer.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>  // Add this include

using namespace vanilla::pricers;

void print_option_results(const OptionData& option, double price, const Greeks& greeks) {
    std::cout << std::fixed << std::setprecision(6)
              << "Option Details:\n"
              << "  Spot: " << option.spot
              << "\n  Strike: " << option.strike
              << "\n  Time to Maturity: " << option.timeToMaturity
              << "\n  Rate: " << option.rateAnnual
              << "\n  Volatility: " << option.volatilityAnnual
              << "\n\nResults:\n"
              << "  Price: " << price
              << "\n  Delta: " << greeks.delta
              << "\n  Gamma: " << greeks.gamma
              << "\n  Vega: " << greeks.vega
              << "\n  Theta: " << greeks.theta
              << std::endl;
}

int main() {
    // Initialize the pricer
    BlackScholesPricer pricer;

    // Single option example
    OptionData option{
        .spot = 100.0,
        .strike = 100.0,
        .timeToMaturity = 1.0,
        .rateAnnual = 0.05,
        .volatilityAnnual = 0.2,
        .optionType = OptionType::CALL  // Updated to use enum
    };

    // Price and compute Greeks
    double price = pricer.price(option);
    Greeks greeks = pricer.computeGreeks(option);

    // Print results
    std::cout << "=== Single Option Example ===\n";
    print_option_results(option, price, greeks);

    // Batch pricing example
    std::cout << "\n=== Batch Pricing Example ===\n";
    const size_t num_options = 1000;
    std::vector<OptionData> options;
    options.reserve(num_options);

    // Create options with different strikes
    for (size_t i = 0; i < num_options; ++i) {
        OptionData batch_option = option;
        batch_option.strike = 80.0 + (i * 40.0 / num_options);
        options.push_back(batch_option);
    }

    // Time the batch pricing
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd prices = pricer.priceVector(options);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Priced " << num_options << " options in "
              << duration.count() << " microseconds\n"
              << "Average time per option: "
              << static_cast<double>(duration.count()) / num_options
              << " microseconds" << std::endl;

    // Print first few results
    std::cout << "\nFirst 5 batch results:\n";
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "Strike: " << options[i].strike
                  << ", Price: " << prices(i) << "\n";
    }

    return 0;
}