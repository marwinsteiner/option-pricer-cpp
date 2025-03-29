"""
Configuration for pytest.

This file contains configuration and fixtures for pytest.
"""

import pytest
import numpy as np


@pytest.fixture
def option_params():
    """
    Fixture providing standard option parameters for tests.
    
    Returns:
        dict: Dictionary containing standard option parameters.
    """
    return {
        'S': 100.0,  # Spot price
        'K': 100.0,  # Strike price
        'T': 1.0,    # Time to maturity (years)
        'r': 0.05,   # Risk-free rate (5%)
        'sigma': 0.2,  # Volatility (20%) - for Black-Scholes
        'sigma_normal': 20.0,  # Absolute volatility for Normal model
        'v0': 0.04,  # Initial variance for stochastic volatility models
    }


@pytest.fixture
def sv_params():
    """
    Fixture providing standard stochastic volatility model parameters for tests.
    
    Returns:
        dict: Dictionary containing standard stochastic volatility model parameters.
    """
    return {
        'kappa_ln': 1.5,   # Mean reversion speed for Log-Normal SV
        'theta_ln': 0.04,  # Long-term volatility for Log-Normal SV
        'sigma_ln': 0.3,   # Volatility of volatility for Log-Normal SV
        'rho_ln': -0.7,    # Correlation for Log-Normal SV
        
        'kappa_h': 2.0,    # Mean reversion speed for Heston
        'theta_h': 0.04,   # Long-term variance for Heston
        'sigma_h': 0.3,    # Volatility of variance for Heston
        'rho_h': -0.7,     # Correlation for Heston
    }


@pytest.fixture
def vector_params():
    """
    Fixture providing vectorized option parameters for tests.
    
    Returns:
        dict: Dictionary containing vectorized option parameters.
    """
    return {
        'S': np.linspace(90, 110, 5),  # Spot prices from 90 to 110
        'K': np.ones(5) * 100.0,       # Strike price of 100
        'T': np.ones(5) * 1.0,         # Time to maturity of 1 year
        'r': np.ones(5) * 0.05,        # Risk-free rate of 5%
        'sigma': np.ones(5) * 0.2,     # Volatility of 20% for Black-Scholes
        'sigma_normal': np.ones(5) * 20.0,  # Absolute volatility for Normal model
    }
