#include "models/normal.hpp"
#include <cmath>

namespace vanilla {
namespace models {

constexpr double PI = 3.141592653589793238463;
constexpr double SQRT_2PI = std::sqrt(2.0 * PI);

double Normal::normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double Normal::normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

double Normal::price_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Bachelier formula for call options
    return std::exp(-r * T) * (
        (F - K) * normal_cdf(d) + 
        stdDev * normal_pdf(d)
    );
}

double Normal::price_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - S, 0.0);
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Bachelier formula for put options
    return std::exp(-r * T) * (
        (K - F) * normal_cdf(-d) + 
        stdDev * normal_pdf(d)
    );
}

Eigen::VectorXd Normal::price_call_vector(
    const Eigen::VectorXd& S,
    const Eigen::VectorXd& K,
    const Eigen::VectorXd& T,
    const Eigen::VectorXd& r,
    const Eigen::VectorXd& sigma
) {
    const size_t n = S.size();
    Eigen::VectorXd prices(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        prices(i) = price_call(S(i), K(i), T(i), r(i), sigma(i));
    }

    return prices;
}

Eigen::VectorXd Normal::price_put_vector(
    const Eigen::VectorXd& S,
    const Eigen::VectorXd& K,
    const Eigen::VectorXd& T,
    const Eigen::VectorXd& r,
    const Eigen::VectorXd& sigma
) {
    const size_t n = S.size();
    Eigen::VectorXd prices(n);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        prices(i) = price_put(S(i), K(i), T(i), r(i), sigma(i));
    }

    return prices;
}

double Normal::delta_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return S >= K ? 1.0 : 0.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Delta for Bachelier model
    return std::exp(-r * T) * normal_cdf(d);
}

double Normal::delta_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return S >= K ? 0.0 : -1.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Delta for Bachelier model (put)
    return std::exp(-r * T) * (normal_cdf(d) - 1.0);
}

double Normal::gamma(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Gamma for Bachelier model
    // Derivative of delta with respect to S
    return std::exp(-r * T) * normal_pdf(d) / (S * stdDev);
}

double Normal::vega(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Vega for Bachelier model
    // Derivative of price with respect to sigma
    return std::exp(-r * T) * std::sqrt(T) * normal_pdf(d);
}

double Normal::theta_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Theta for Bachelier model (call)
    double theta = -sigma * std::exp(-r * T) * normal_pdf(d) / (2.0 * std::sqrt(T)) 
                  - r * std::exp(-r * T) * (F - K) * normal_cdf(d)
                  - r * std::exp(-r * T) * stdDev * normal_pdf(d);
    
    return theta;
}

double Normal::theta_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;
    
    // Forward price
    double F = S * std::exp(r * T);
    double stdDev = sigma * std::sqrt(T);
    double d = (F - K) / stdDev;
    
    // Theta for Bachelier model (put)
    double theta = -sigma * std::exp(-r * T) * normal_pdf(d) / (2.0 * std::sqrt(T)) 
                  - r * std::exp(-r * T) * (K - F) * normal_cdf(-d)
                  - r * std::exp(-r * T) * stdDev * normal_pdf(d);
    
    return theta;
}

} // namespace models
} // namespace vanilla
