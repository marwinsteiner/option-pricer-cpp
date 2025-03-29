#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "models/black_scholes.hpp"
#include "models/normal.hpp"
#include "models/log_normal_sv.hpp"
#include "models/heston_sv.hpp"
#include "pricers/black_scholes_pricer.hpp"

namespace py = pybind11;
using namespace vanilla::models;
using namespace vanilla::pricers;

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ bindings for option pricing models";
    
    // Enum for option types
    py::enum_<OptionType>(m, "OptionType")
        .value("CALL", OptionType::CALL)
        .value("PUT", OptionType::PUT)
        .value("INVERSE_CALL", OptionType::INVERSE_CALL)
        .value("INVERSE_PUT", OptionType::INVERSE_PUT)
        .export_values();
    
    // Bind OptionData structure
    py::class_<OptionData>(m, "OptionData")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double, OptionType>(),
             py::arg("spot"), py::arg("strike"), py::arg("timeToMaturity"),
             py::arg("rateAnnual"), py::arg("volatilityAnnual"), py::arg("optionType") = OptionType::CALL)
        .def_readwrite("spot", &OptionData::spot)
        .def_readwrite("strike", &OptionData::strike)
        .def_readwrite("timeToMaturity", &OptionData::timeToMaturity)
        .def_readwrite("rateAnnual", &OptionData::rateAnnual)
        .def_readwrite("volatilityAnnual", &OptionData::volatilityAnnual)
        .def_readwrite("optionType", &OptionData::optionType);
    
    // Bind Greeks structure
    py::class_<Greeks>(m, "Greeks")
        .def(py::init<>())
        .def_readwrite("delta", &Greeks::delta)
        .def_readwrite("gamma", &Greeks::gamma)
        .def_readwrite("vega", &Greeks::vega)
        .def_readwrite("theta", &Greeks::theta);
    
    // Bind BlackScholes model
    py::class_<BlackScholes>(m, "BlackScholes")
        .def(py::init<>())
        .def_static("price_call", &BlackScholes::price_call)
        .def_static("price_put", &BlackScholes::price_put)
        .def_static("delta_call", &BlackScholes::delta_call)
        .def_static("delta_put", &BlackScholes::delta_put)
        .def_static("gamma", &BlackScholes::gamma)
        .def_static("vega", &BlackScholes::vega)
        .def_static("theta_call", &BlackScholes::theta_call)
        .def_static("theta_put", &BlackScholes::theta_put)
        // Add implied volatility methods that use the pricer internally
        .def_static("implied_volatility_call", [](double price, double S, double K, double T, double r,
                                                double initial_guess, int max_iterations, double tolerance) {
            // Create option data for a call option
            OptionData option;
            option.spot = S;
            option.strike = K;
            option.timeToMaturity = T;
            option.rateAnnual = r;
            option.volatilityAnnual = initial_guess;  // Initial guess
            option.optionType = OptionType::CALL;
            
            // Create a pricer and compute implied vol
            BlackScholesPricer pricer;
            return pricer.impliedVolatility(option, price, initial_guess, tolerance, max_iterations);
        }, py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
           py::arg("initial_guess") = 0.2, py::arg("max_iterations") = 100, py::arg("tolerance") = 1e-8)
        .def_static("implied_volatility_put", [](double price, double S, double K, double T, double r,
                                               double initial_guess, int max_iterations, double tolerance) {
            // Create option data for a put option
            OptionData option;
            option.spot = S;
            option.strike = K;
            option.timeToMaturity = T;
            option.rateAnnual = r;
            option.volatilityAnnual = initial_guess;  // Initial guess
            option.optionType = OptionType::PUT;
            
            // Create a pricer and compute implied vol
            BlackScholesPricer pricer;
            return pricer.impliedVolatility(option, price, initial_guess, tolerance, max_iterations);
        }, py::arg("price"), py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"),
           py::arg("initial_guess") = 0.2, py::arg("max_iterations") = 100, py::arg("tolerance") = 1e-8)
        .def_static("price_call_vector", [](py::array_t<double> S, py::array_t<double> K,
                                          py::array_t<double> T, py::array_t<double> r,
                                          py::array_t<double> sigma) {
            // Convert numpy arrays to Eigen::VectorXd
            Eigen::Map<const Eigen::VectorXd> S_vec(S.data(), S.size());
            Eigen::Map<const Eigen::VectorXd> K_vec(K.data(), K.size());
            Eigen::Map<const Eigen::VectorXd> T_vec(T.data(), T.size());
            Eigen::Map<const Eigen::VectorXd> r_vec(r.data(), r.size());
            Eigen::Map<const Eigen::VectorXd> sigma_vec(sigma.data(), sigma.size());
            
            // Call the C++ function
            Eigen::VectorXd result = BlackScholes::price_call_vector(S_vec, K_vec, T_vec, r_vec, sigma_vec);
            
            // Convert the result back to a numpy array
            return py::array_t<double>(result.size(), result.data());
        })
        .def_static("price_put_vector", [](py::array_t<double> S, py::array_t<double> K,
                                         py::array_t<double> T, py::array_t<double> r,
                                         py::array_t<double> sigma) {
            // Convert numpy arrays to Eigen::VectorXd
            Eigen::Map<const Eigen::VectorXd> S_vec(S.data(), S.size());
            Eigen::Map<const Eigen::VectorXd> K_vec(K.data(), K.size());
            Eigen::Map<const Eigen::VectorXd> T_vec(T.data(), T.size());
            Eigen::Map<const Eigen::VectorXd> r_vec(r.data(), r.size());
            Eigen::Map<const Eigen::VectorXd> sigma_vec(sigma.data(), sigma.size());
            
            // Call the C++ function
            Eigen::VectorXd result = BlackScholes::price_put_vector(S_vec, K_vec, T_vec, r_vec, sigma_vec);
            
            // Convert the result back to a numpy array
            return py::array_t<double>(result.size(), result.data());
        });
    
    // Bind Normal (Bachelier) model
    py::class_<Normal>(m, "Normal")
        .def(py::init<>())
        .def_static("price_call", &Normal::price_call)
        .def_static("price_put", &Normal::price_put)
        .def_static("delta_call", &Normal::delta_call)
        .def_static("delta_put", &Normal::delta_put)
        .def_static("gamma", &Normal::gamma)
        .def_static("vega", &Normal::vega)
        .def_static("theta_call", &Normal::theta_call)
        .def_static("theta_put", &Normal::theta_put)
        .def_static("price_call_vector", [](py::array_t<double> S, py::array_t<double> K,
                                          py::array_t<double> T, py::array_t<double> r,
                                          py::array_t<double> sigma) {
            // Convert numpy arrays to Eigen::VectorXd
            Eigen::Map<const Eigen::VectorXd> S_vec(S.data(), S.size());
            Eigen::Map<const Eigen::VectorXd> K_vec(K.data(), K.size());
            Eigen::Map<const Eigen::VectorXd> T_vec(T.data(), T.size());
            Eigen::Map<const Eigen::VectorXd> r_vec(r.data(), r.size());
            Eigen::Map<const Eigen::VectorXd> sigma_vec(sigma.data(), sigma.size());
            
            // Call the C++ function
            Eigen::VectorXd result = Normal::price_call_vector(S_vec, K_vec, T_vec, r_vec, sigma_vec);
            
            // Convert the result back to a numpy array
            return py::array_t<double>(result.size(), result.data());
        })
        .def_static("price_put_vector", [](py::array_t<double> S, py::array_t<double> K,
                                         py::array_t<double> T, py::array_t<double> r,
                                         py::array_t<double> sigma) {
            // Convert numpy arrays to Eigen::VectorXd
            Eigen::Map<const Eigen::VectorXd> S_vec(S.data(), S.size());
            Eigen::Map<const Eigen::VectorXd> K_vec(K.data(), K.size());
            Eigen::Map<const Eigen::VectorXd> T_vec(T.data(), T.size());
            Eigen::Map<const Eigen::VectorXd> r_vec(r.data(), r.size());
            Eigen::Map<const Eigen::VectorXd> sigma_vec(sigma.data(), sigma.size());
            
            // Call the C++ function
            Eigen::VectorXd result = Normal::price_put_vector(S_vec, K_vec, T_vec, r_vec, sigma_vec);
            
            // Convert the result back to a numpy array
            return py::array_t<double>(result.size(), result.data());
        });
    
    // Define a Python class for PathSimulation
    py::class_<StochasticVolModel::PathSimulation>(m, "PathSimulation")
        .def(py::init<>())
        .def_readwrite("timePoints", &StochasticVolModel::PathSimulation::timePoints)
        .def_readwrite("spotPaths", &StochasticVolModel::PathSimulation::spotPaths)
        .def_readwrite("volPaths", &StochasticVolModel::PathSimulation::volPaths);
    
    // Bind LogNormalSV model
    py::class_<LogNormalSV>(m, "LogNormalSV")
        .def(py::init<double, double, double, double>(),
             py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"))
        .def("price_european_call", &LogNormalSV::priceEuropeanCall,
             py::arg("S"), py::arg("K"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("price_european_put", &LogNormalSV::priceEuropeanPut,
             py::arg("S"), py::arg("K"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("characteristic_function", &LogNormalSV::characteristicFunction,
             py::arg("u"), py::arg("S"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("simulate_paths", &LogNormalSV::simulatePaths,
             py::arg("S0"), py::arg("v0"), py::arg("T"), py::arg("r"),
             py::arg("numPaths") = 10000, py::arg("numSteps") = 252);
    
    // Bind HestonSV model
    py::class_<HestonSV>(m, "HestonSV")
        .def(py::init<double, double, double, double, double>(),
             py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("v0"))
        .def("price_european_call", &HestonSV::priceEuropeanCall,
             py::arg("S"), py::arg("K"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("price_european_put", &HestonSV::priceEuropeanPut,
             py::arg("S"), py::arg("K"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("characteristic_function", &HestonSV::characteristicFunction,
             py::arg("u"), py::arg("S"), py::arg("v0"), py::arg("T"), py::arg("r"))
        .def("simulate_paths", &HestonSV::simulatePaths,
             py::arg("S0"), py::arg("v0"), py::arg("T"), py::arg("r"),
             py::arg("numPaths") = 10000, py::arg("numSteps") = 252);
}
