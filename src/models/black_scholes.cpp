#include "models/black_scholes.hpp"
#include <cmath>

namespace vanilla {
namespace models {

constexpr double PI = 3.141592653589793238463;
constexpr double SQRT_2PI = std::sqrt(2.0 * PI);

double BlackScholes::normal_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double BlackScholes::normal_pdf(double x) {
    return std::exp(-0.5 * x * x) / SQRT_2PI;
}

void BlackScholes::compute_d1_d2(double S, double K, double T, double r, double sigma,
                                double& d1, double& d2) {
    d1 = (std::log(S/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    d2 = d1 - sigma * std::sqrt(T);
}

double BlackScholes::price_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);

    return S * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
}

double BlackScholes::price_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(K - S, 0.0);

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);

    return K * std::exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);
}

Eigen::VectorXd BlackScholes::price_call_vector(
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

Eigen::VectorXd BlackScholes::price_put_vector(
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

double BlackScholes::delta_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return S >= K ? 1.0 : 0.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);
    return normal_cdf(d1);
}

double BlackScholes::delta_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return S >= K ? 0.0 : -1.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);
    return normal_cdf(d1) - 1.0;
}

double BlackScholes::gamma(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);
    return normal_pdf(d1) / (S * sigma * std::sqrt(T));
}

double BlackScholes::vega(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);
    return S * std::sqrt(T) * normal_pdf(d1);
}

double BlackScholes::theta_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);

    double theta = -(S * sigma * normal_pdf(d1)) / (2.0 * std::sqrt(T))
                  - r * K * std::exp(-r * T) * normal_cdf(d2);
    return theta;
}

double BlackScholes::theta_put(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return 0.0;

    double d1, d2;
    compute_d1_d2(S, K, T, r, sigma, d1, d2);

    double theta = -(S * sigma * normal_pdf(d1)) / (2.0 * std::sqrt(T))
                  + r * K * std::exp(-r * T) * normal_cdf(-d2);
    return theta;
}

} // namespace models
} // namespace vanilla