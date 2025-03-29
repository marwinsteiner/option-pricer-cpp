"""
Tests for the Normal (Bachelier) model.

This module contains tests for the Normal (Bachelier) option pricing model.
"""

import pytest
import numpy as np
from option_pricer import Normal


def test_normal_call_price(option_params):
    """Test Normal (Bachelier) call option pricing."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # When
    call_price = Normal.price_call(S, K, T, r, sigma)
    
    # Then
    expected_price = 10.276275
    assert abs(call_price - expected_price) < 1e-6


def test_normal_put_price(option_params):
    """Test Normal (Bachelier) put option pricing."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # When
    put_price = Normal.price_put(S, K, T, r, sigma)
    
    # Then
    expected_price = 5.398217
    assert abs(put_price - expected_price) < 1e-6


def test_normal_put_call_parity(option_params):
    """Test put-call parity for Normal (Bachelier) model."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # When
    call_price = Normal.price_call(S, K, T, r, sigma)
    put_price = Normal.price_put(S, K, T, r, sigma)
    
    # Then
    # Put-call parity: C - P = S - K * exp(-r * T)
    parity_value = call_price - put_price
    expected_value = S - K * np.exp(-r * T)
    assert abs(parity_value - expected_value) < 1e-6


def test_normal_greeks(option_params):
    """Test Normal (Bachelier) Greeks calculations."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # When
    delta_call = Normal.delta_call(S, K, T, r, sigma)
    delta_put = Normal.delta_put(S, K, T, r, sigma)
    gamma = Normal.gamma(S, K, T, r, sigma)
    vega = Normal.vega(S, K, T, r, sigma)
    theta_call = Normal.theta_call(S, K, T, r, sigma)
    theta_put = Normal.theta_put(S, K, T, r, sigma)
    
    # Then
    expected_delta_call = 0.571843
    expected_delta_put = expected_delta_call - 1.0
    expected_gamma = 0.000184
    expected_vega = 0.367219
    expected_theta_call = -4.186001
    
    assert abs(delta_call - expected_delta_call) < 1e-6
    assert abs(delta_put - expected_delta_put) < 1e-6
    assert abs(gamma - expected_gamma) < 1e-6
    assert abs(vega - expected_vega) < 1e-6
    assert abs(theta_call - expected_theta_call) < 1e-6


def test_normal_atm_option(option_params):
    """Test Normal (Bachelier) model for at-the-money options."""
    # Given
    S = option_params['S']
    K = option_params['K']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # When
    call_price = Normal.price_call(S, K, T, r, sigma)
    put_price = Normal.price_put(S, K, T, r, sigma)
    
    # Then
    # For ATM options with forward = strike, call and put should have same time value
    forward = S * np.exp(r * T)
    if abs(forward - K) < 1e-10:
        intrinsic_call = max(0, forward - K) * np.exp(-r * T)
        intrinsic_put = max(0, K - forward) * np.exp(-r * T)
        
        time_value_call = call_price - intrinsic_call
        time_value_put = put_price - intrinsic_put
        
        assert abs(time_value_call - time_value_put) < 1e-6


def test_normal_deep_itm_otm(option_params):
    """Test Normal (Bachelier) model for deep in-the-money and out-of-the-money options."""
    # Given
    S = option_params['S']
    T = option_params['T']
    r = option_params['r']
    sigma = option_params['sigma_normal']  # Use absolute volatility
    
    # Deep ITM call / OTM put (K = 50)
    K_itm = 50.0
    
    # Deep OTM call / ITM put (K = 150)
    K_otm = 150.0
    
    # When
    itm_call = Normal.price_call(S, K_itm, T, r, sigma)
    otm_call = Normal.price_call(S, K_otm, T, r, sigma)
    itm_put = Normal.price_put(S, K_otm, T, r, sigma)
    otm_put = Normal.price_put(S, K_itm, T, r, sigma)
    
    # Then
    # ITM call should be approximately S - K_itm * exp(-r * T) for deep ITM
    expected_itm_call = S - K_itm * np.exp(-r * T)
    assert abs(itm_call - expected_itm_call) / expected_itm_call < 0.1
    
    # OTM call should be small but positive
    assert otm_call > 0 and otm_call < S * 0.1
    
    # ITM put should be approximately K_otm * exp(-r * T) - S for deep ITM
    expected_itm_put = K_otm * np.exp(-r * T) - S
    assert abs(itm_put - expected_itm_put) / expected_itm_put < 0.1
    
    # OTM put should be small but positive
    assert otm_put > 0 and otm_put < K_itm * 0.1


def test_normal_vectorized(vector_params):
    """Test vectorized operations for Normal (Bachelier) model."""
    # Given
    S = vector_params['S']
    K = vector_params['K']
    T = vector_params['T']
    r = vector_params['r']
    sigma = vector_params['sigma_normal']  # Use absolute volatility
    
    # When
    call_prices = Normal.price_call(S, K, T, r, sigma)
    put_prices = Normal.price_put(S, K, T, r, sigma)
    
    # Then
    assert len(call_prices) == len(S)
    assert len(put_prices) == len(S)
    
    # Test individual prices
    for i in range(len(S)):
        expected_call = Normal.price_call(S[i], K[i], T[i], r[i], sigma[i])
        expected_put = Normal.price_put(S[i], K[i], T[i], r[i], sigma[i])
        
        assert abs(call_prices[i] - expected_call) < 1e-6
        assert abs(put_prices[i] - expected_put) < 1e-6
    
    # Test put-call parity for all prices
    parity_values = call_prices - put_prices
    expected_values = S - K * np.exp(-r * T)
    
    for i in range(len(S)):
        assert abs(parity_values[i] - expected_values[i]) < 1e-6
