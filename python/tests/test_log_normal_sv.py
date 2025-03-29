"""
Tests for the Log-Normal Stochastic Volatility model.

This module contains tests for the Log-Normal Stochastic Volatility option pricing model.
"""

import pytest
import numpy as np
from option_pricer import LogNormalSV, BlackScholes


def test_log_normal_sv_initialization(sv_params):
    """Test initialization of Log-Normal SV model."""
    # Given
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    # When
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # Then
    assert model is not None


def test_log_normal_sv_call_price(option_params, sv_params):
    """Test Log-Normal SV call option pricing."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # When
    call_price = model.price_european_call(S, K, v0, T, r)
    
    # Then
    expected_price = 10.448993  # Expected price from reference implementation
    assert abs(call_price - expected_price) < 1e-6


def test_log_normal_sv_put_price(option_params, sv_params):
    """Test Log-Normal SV put option pricing."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # When
    put_price = model.price_european_put(S, K, v0, T, r)
    
    # Then
    expected_price = 5.571936  # Expected price from reference implementation
    assert abs(put_price - expected_price) < 1e-6


def test_log_normal_sv_put_call_parity(option_params, sv_params):
    """Test put-call parity for Log-Normal SV model."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # When
    call_price = model.price_european_call(S, K, v0, T, r)
    put_price = model.price_european_put(S, K, v0, T, r)
    
    # Then
    # Put-call parity: C - P = S - K * exp(-r * T)
    parity_value = call_price - put_price
    expected_value = S - K * np.exp(-r * T)
    assert abs(parity_value - expected_value) < 1e-6


def test_log_normal_sv_vs_black_scholes(option_params, sv_params):
    """Test Log-Normal SV model vs Black-Scholes for zero vol of vol."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    sigma_bs = np.sqrt(v0)  # Volatility for Black-Scholes
    
    # Create a Log-Normal SV model with zero vol of vol and zero correlation
    # This should behave like Black-Scholes
    kappa = 0.0  # No mean reversion
    theta = v0   # Long-term vol equals initial vol
    sigma = 0.0  # Zero vol of vol
    rho = 0.0    # Zero correlation
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # When
    ln_call_price = model.price_european_call(S, K, v0, T, r)
    ln_put_price = model.price_european_put(S, K, v0, T, r)
    
    bs_call_price = BlackScholes.price_call(S, K, T, r, sigma_bs)
    bs_put_price = BlackScholes.price_put(S, K, T, r, sigma_bs)
    
    # Then
    # Prices should be close (not exactly equal due to different implementations)
    assert abs(ln_call_price - bs_call_price) / bs_call_price < 0.05
    assert abs(ln_put_price - bs_put_price) / bs_put_price < 0.05


def test_log_normal_sv_characteristic_function(option_params, sv_params):
    """Test Log-Normal SV characteristic function."""
    # Given
    S = option_params['S']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    
    # When
    # Test characteristic function at u = 0 (should equal 1)
    cf_at_zero = model.characteristic_function(0j, S, v0, T, r)
    
    # Test characteristic function at u = -i (should equal exp(r*T))
    cf_at_minus_i = model.characteristic_function(-1j, S, v0, T, r)
    
    # Then
    assert abs(cf_at_zero - 1.0) < 1e-6
    assert abs(cf_at_minus_i - np.exp(r * T)) < 1e-6


def test_log_normal_sv_simulation(option_params, sv_params):
    """Test Log-Normal SV path simulation."""
    # Given
    S = option_params['S']
    T = option_params['T']
    r = option_params['r']
    v0 = option_params['v0']
    
    kappa = sv_params['kappa_ln']
    theta = sv_params['theta_ln']
    sigma = sv_params['sigma_ln']
    rho = sv_params['rho_ln']
    
    model = LogNormalSV(kappa, theta, sigma, rho)
    num_paths = 100
    num_steps = 252
    
    # When
    paths = model.simulate_paths(S, v0, T, r, num_paths, num_steps)
    
    # Then
    # Check dimensions
    assert paths['time_points'].shape == (num_steps + 1,)
    assert paths['spot_paths'].shape == (num_steps + 1, num_paths)
    assert paths['vol_paths'].shape == (num_steps + 1, num_paths)
    
    # Check initial values
    assert np.all(paths['spot_paths'][0, :] == S)
    assert np.all(abs(paths['vol_paths'][0, :] - np.sqrt(v0)) < 1e-6)
    
    # Check time points
    assert paths['time_points'][0] == 0.0
    assert abs(paths['time_points'][-1] - T) < 1e-6
    
    # Check average final spot (should be close to S * exp(r * T) for risk-neutral simulation)
    expected_avg_spot = S * np.exp(r * T)
    actual_avg_spot = np.mean(paths['spot_paths'][-1, :])
    
    # Allow for some Monte Carlo error (within 5%)
    assert abs(actual_avg_spot - expected_avg_spot) / expected_avg_spot < 0.05
