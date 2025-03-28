#include "pricers/black_scholes_pricer.hpp"
#include "pricers/normal_pricer.hpp"
#include "models/log_normal_sv.hpp"
#include "models/heston_sv.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

using namespace vanilla::pricers;
using namespace vanilla::models;

// Helper function to print option results
void print_option_results(const std::string& model_name, const OptionData& option, double price) {
    std::cout << std::fixed << std::setprecision(6)
              << "=== " << model_name << " Model Results ===\n"
              << "Option Details:\n"
              << "  Type: " << (option.optionType == OptionType::CALL ? "Call" : 
                               option.optionType == OptionType::PUT ? "Put" :
                               option.optionType == OptionType::INVERSE_CALL ? "Inverse Call" : "Inverse Put")
              << "\n  Spot: " << option.spot
              << "\n  Strike: " << option.strike
              << "\n  Time to Maturity: " << option.timeToMaturity
              << "\n  Rate: " << option.rateAnnual
              << "\n  Volatility: " << option.volatilityAnnual
              << "\n\nPrice: " << price
              << "\n" << std::string(40, '-') << "\n" << std::endl;
}

// Helper function to print Greeks
void print_greeks(const std::string& model_name, const Greeks& greeks) {
    std::cout << std::fixed << std::setprecision(6)
              << model_name << " Greeks:\n"
              << "  Delta: " << greeks.delta
              << "\n  Gamma: " << greeks.gamma
              << "\n  Vega: " << greeks.vega
              << "\n  Theta: " << greeks.theta
              << "\n" << std::string(40, '-') << "\n" << std::endl;
}

// Helper function to print Monte Carlo simulation results
void print_mc_results(const std::string& model_name, 
                     const StochasticVolModel::PathSimulation& simulation,
                     int num_paths) {
    // Print first and last time points
    std::cout << std::fixed << std::setprecision(6)
              << "=== " << model_name << " Monte Carlo Simulation ===\n"
              << "Number of paths: " << num_paths << "\n"
              << "Time points: " << simulation.timePoints.size() << "\n\n"
              << "First 3 paths (first and last time points):\n";
    
    // Print header
    std::cout << "Path | Initial Spot | Final Spot | Initial Vol | Final Vol\n"
              << std::string(60, '-') << "\n";
    
    // Print first 3 paths
    for (int i = 0; i < std::min(3, num_paths); ++i) {
        int last_idx = simulation.timePoints.size() - 1;
        std::cout << " " << i + 1 << "   | "
                  << std::setw(12) << simulation.spotPaths(0, i) << " | "
                  << std::setw(10) << simulation.spotPaths(last_idx, i) << " | "
                  << std::setw(11) << simulation.volPaths(0, i) << " | "
                  << std::setw(9) << simulation.volPaths(last_idx, i) << "\n";
    }
    
    // Print averages
    double avg_final_spot = 0.0;
    double avg_final_vol = 0.0;
    int last_idx = simulation.timePoints.size() - 1;
    
    for (int i = 0; i < num_paths; ++i) {
        avg_final_spot += simulation.spotPaths(last_idx, i);
        avg_final_vol += simulation.volPaths(last_idx, i);
    }
    
    avg_final_spot /= num_paths;
    avg_final_vol /= num_paths;
    
    std::cout << "\nAverage final spot: " << avg_final_spot
              << "\nAverage final volatility: " << avg_final_vol
              << "\n" << std::string(60, '-') << "\n" << std::endl;
}

int main() {
    std::cout << "=== Advanced Option Pricing Models Example ===\n" << std::endl;
    
    // Define a standard option for comparison across models
    OptionData option{
        .spot = 100.0,
        .strike = 100.0,
        .timeToMaturity = 1.0,
        .rateAnnual = 0.05,
        .volatilityAnnual = 0.2,
        .optionType = OptionType::CALL
    };
    
    // 1. Black-Scholes Model
    std::cout << "1. BLACK-SCHOLES MODEL\n" << std::string(30, '=') << std::endl;
    BlackScholesPricer bs_pricer;
    double bs_price = bs_pricer.price(option);
    Greeks bs_greeks = bs_pricer.computeGreeks(option);
    
    print_option_results("Black-Scholes", option, bs_price);
    print_greeks("Black-Scholes", bs_greeks);
    
    // 2. Normal (Bachelier) Model
    std::cout << "2. NORMAL (BACHELIER) MODEL\n" << std::string(30, '=') << std::endl;
    NormalPricer normal_pricer;
    
    // For Normal model, we need to adjust the volatility to be absolute rather than relative
    // For a 20% volatility on a spot of 100, the absolute volatility should be 20
    double normal_vol = option.spot * option.volatilityAnnual;
    
    // Create a new option data with adjusted volatility for Normal model
    OptionData normal_option = option;
    normal_option.volatilityAnnual = normal_vol;
    
    double normal_price = normal_pricer.price(normal_option);
    Greeks normal_greeks = normal_pricer.computeGreeks(normal_option);
    
    print_option_results("Normal (Bachelier)", normal_option, normal_price);
    print_greeks("Normal (Bachelier)", normal_greeks);
    
    // 3. Log-Normal Stochastic Volatility Model
    std::cout << "3. LOG-NORMAL STOCHASTIC VOLATILITY MODEL\n" << std::string(40, '=') << std::endl;
    
    // Initialize model parameters
    double kappa_ln = 1.5;    // Mean reversion speed
    double theta_ln = 0.04;   // Long-term volatility (squared)
    double sigma_ln = 0.3;    // Volatility of volatility
    double rho_ln = -0.7;     // Correlation between spot and vol
    double v0_ln = 0.04;      // Initial variance (volatility squared)
    
    LogNormalSV ln_sv_model(kappa_ln, theta_ln, sigma_ln, rho_ln);
    
    // Price using Fourier transform
    double ln_sv_call_price = ln_sv_model.priceEuropeanCall(
        option.spot, option.strike, v0_ln, option.timeToMaturity, option.rateAnnual
    );
    
    double ln_sv_put_price = ln_sv_model.priceEuropeanPut(
        option.spot, option.strike, v0_ln, option.timeToMaturity, option.rateAnnual
    );
    
    std::cout << "Log-Normal SV Model Parameters:\n"
              << "  Kappa (mean reversion): " << kappa_ln << "\n"
              << "  Theta (long-term vol): " << std::sqrt(theta_ln) << "\n"
              << "  Sigma (vol of vol): " << sigma_ln << "\n"
              << "  Rho (correlation): " << rho_ln << "\n"
              << "  Initial volatility: " << std::sqrt(v0_ln) << "\n\n"
              << "Fourier Transform Pricing:\n"
              << "  Call Price: " << ln_sv_call_price << "\n"
              << "  Put Price: " << ln_sv_put_price << "\n" 
              << std::string(40, '-') << "\n" << std::endl;
    
    // Monte Carlo simulation
    int num_paths = 10000;
    int num_steps = 252;  // Daily steps for a year
    
    auto start = std::chrono::high_resolution_clock::now();
    
    StochasticVolModel::PathSimulation ln_simulation = ln_sv_model.simulatePaths(
        option.spot, v0_ln, option.timeToMaturity, option.rateAnnual, num_paths, num_steps
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Monte Carlo simulation completed in " 
              << duration.count() << " milliseconds.\n" << std::endl;
    
    print_mc_results("Log-Normal SV", ln_simulation, num_paths);
    
    // 4. Heston Stochastic Volatility Model
    std::cout << "4. HESTON STOCHASTIC VOLATILITY MODEL\n" << std::string(40, '=') << std::endl;
    
    // Initialize model parameters
    double kappa_h = 2.0;    // Mean reversion speed
    double theta_h = 0.04;   // Long-term variance
    double sigma_h = 0.3;    // Volatility of variance
    double rho_h = -0.7;     // Correlation between spot and variance
    double v0_h = 0.04;      // Initial variance
    
    HestonSV heston_model(kappa_h, theta_h, sigma_h, rho_h, v0_h);
    
    // Price using Fourier transform
    double heston_call_price = heston_model.priceEuropeanCall(
        option.spot, option.strike, v0_h, option.timeToMaturity, option.rateAnnual
    );
    
    double heston_put_price = heston_model.priceEuropeanPut(
        option.spot, option.strike, v0_h, option.timeToMaturity, option.rateAnnual
    );
    
    std::cout << "Heston Model Parameters:\n"
              << "  Kappa (mean reversion): " << kappa_h << "\n"
              << "  Theta (long-term variance): " << theta_h << "\n"
              << "  Sigma (vol of variance): " << sigma_h << "\n"
              << "  Rho (correlation): " << rho_h << "\n"
              << "  Initial variance: " << v0_h << "\n\n"
              << "Fourier Transform Pricing:\n"
              << "  Call Price: " << heston_call_price << "\n"
              << "  Put Price: " << heston_put_price << "\n" 
              << std::string(40, '-') << "\n" << std::endl;
    
    // Monte Carlo simulation
    start = std::chrono::high_resolution_clock::now();
    
    StochasticVolModel::PathSimulation heston_simulation = heston_model.simulatePaths(
        option.spot, v0_h, option.timeToMaturity, option.rateAnnual, num_paths, num_steps
    );
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Monte Carlo simulation completed in " 
              << duration.count() << " milliseconds.\n" << std::endl;
    
    print_mc_results("Heston SV", heston_simulation, num_paths);
    
    // 5. Model Comparison
    std::cout << "5. MODEL COMPARISON\n" << std::string(30, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(6)
              << "Call Option Prices:\n"
              << "  Black-Scholes: " << bs_price << "\n"
              << "  Normal (Bachelier): " << normal_price << "\n"
              << "  Log-Normal SV: " << ln_sv_call_price << "\n"
              << "  Heston SV: " << heston_call_price << "\n\n"
              << "Put Option Prices:\n"
              << "  Black-Scholes: " << bs_pricer.price({option.spot, option.strike, option.timeToMaturity, 
                                                        option.rateAnnual, option.volatilityAnnual, OptionType::PUT}) << "\n"
              << "  Normal (Bachelier): " << normal_pricer.price({option.spot, option.strike, option.timeToMaturity, 
                                                                option.rateAnnual, normal_vol, OptionType::PUT}) << "\n"
              << "  Log-Normal SV: " << ln_sv_put_price << "\n"
              << "  Heston SV: " << heston_put_price << std::endl;
    
    return 0;
}
