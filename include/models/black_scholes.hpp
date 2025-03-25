#pragma once

#include <cmath>
#include <Eigen/Dense>

namespace vanilla {
    namespace models {

        class BlackScholes {
        public:
            // Single option pricing
            static double price_call(double S, double K, double T, double r, double sigma);
            static double price_put(double S, double K, double T, double r, double sigma);

            // Vectorized pricing
            static Eigen::VectorXd price_call_vector(
                const Eigen::VectorXd& S,
                const Eigen::VectorXd& K,
                const Eigen::VectorXd& T,
                const Eigen::VectorXd& r,
                const Eigen::VectorXd& sigma
            );

            static Eigen::VectorXd price_put_vector(
                const Eigen::VectorXd& S,
                const Eigen::VectorXd& K,
                const Eigen::VectorXd& T,
                const Eigen::VectorXd& r,
                const Eigen::VectorXd& sigma
            );

            // Greeks
            static double delta_call(double S, double K, double T, double r, double sigma);
            static double delta_put(double S, double K, double T, double r, double sigma);
            static double gamma(double S, double K, double T, double r, double sigma);
            static double vega(double S, double K, double T, double r, double sigma);
            static double theta_call(double S, double K, double T, double r, double sigma);
            static double theta_put(double S, double K, double T, double r, double sigma);

        private:
            static double normal_cdf(double x);
            static double normal_pdf(double x);
            static void compute_d1_d2(double S, double K, double T, double r, double sigma,
                                    double& d1, double& d2);
        };

    } // namespace models
} // namespace vanilla