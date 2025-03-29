# Option Pricer

A Python package for option pricing using various models. This package is a Python binding for a C++ implementation of Artur Sepp's option pricing models.

## Features

- Black-Scholes model
- Normal (Bachelier) model
- Log-Normal Stochastic Volatility model
- Heston Stochastic Volatility model
- Vectorized pricing using NumPy arrays
- Monte Carlo simulation for stochastic volatility models

## Installation

```bash
pip install option-pricer
```

## Requirements

- Python 3.6+
- NumPy
- A C++ compiler (GCC, MSVC, Clang)
- CMake 3.10+

## Usage Examples

### Black-Scholes Model

```python
import numpy as np
from option_pricer import BlackScholes

# Price a call option
S = 100  # Spot price
K = 100  # Strike price
T = 1.0  # Time to maturity (years)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility (20%)

call_price = BlackScholes.price_call(S, K, T, r, sigma)
print(f"Black-Scholes Call Price: {call_price:.6f}")  # 10.450584

# Price a put option
put_price = BlackScholes.price_put(S, K, T, r, sigma)
print(f"Black-Scholes Put Price: {put_price:.6f}")  # 5.573526

# Calculate Greeks
delta = BlackScholes.delta_call(S, K, T, r, sigma)
gamma = BlackScholes.gamma(S, K, T, r, sigma)
vega = BlackScholes.vega(S, K, T, r, sigma)
theta = BlackScholes.theta_call(S, K, T, r, sigma)

print(f"Delta: {delta:.6f}")  # 0.636831
print(f"Gamma: {gamma:.6f}")  # 0.018762
print(f"Vega: {vega:.6f}")    # 37.524035
print(f"Theta: {theta:.6f}")  # -6.414028

# Vectorized pricing
spots = np.linspace(90, 110, 21)
call_prices = BlackScholes.price_call(spots, K, T, r, sigma)
print(f"Call prices for spots from 90 to 110: {call_prices}")
```

### Normal (Bachelier) Model

```python
from option_pricer import Normal

# Note: For Normal model, volatility should be absolute, not relative
# For a 20% volatility on a spot of 100, the absolute volatility is 20
S = 100
K = 100
T = 1.0
r = 0.05
sigma = 20  # Absolute volatility

call_price = Normal.price_call(S, K, T, r, sigma)
print(f"Normal (Bachelier) Call Price: {call_price:.6f}")  # 10.276275

put_price = Normal.price_put(S, K, T, r, sigma)
print(f"Normal (Bachelier) Put Price: {put_price:.6f}")  # 5.398217
```

### Stochastic Volatility Models

```python
from option_pricer import LogNormalSV, HestonSV
import matplotlib.pyplot as plt

# Log-Normal SV model
kappa_ln = 1.5    # Mean reversion speed
theta_ln = 0.04   # Long-term volatility (squared)
sigma_ln = 0.3    # Volatility of volatility
rho_ln = -0.7     # Correlation between spot and vol
v0_ln = 0.04      # Initial variance (volatility squared)

ln_sv_model = LogNormalSV(kappa_ln, theta_ln, sigma_ln, rho_ln)

# Price European options
ln_call_price = ln_sv_model.price_european_call(S, K, v0_ln, T, r)
ln_put_price = ln_sv_model.price_european_put(S, K, v0_ln, T, r)

print(f"Log-Normal SV Call Price: {ln_call_price:.6f}")  # 10.448993
print(f"Log-Normal SV Put Price: {ln_put_price:.6f}")    # 5.571936

# Simulate paths
ln_paths = ln_sv_model.simulate_paths(S, v0_ln, T, r, num_paths=100, num_steps=252)

# Plot a few paths
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for i in range(5):
    plt.plot(ln_paths['time_points'], ln_paths['spot_paths'][:, i])
plt.title('Log-Normal SV - Spot Price Paths')
plt.xlabel('Time')
plt.ylabel('Spot Price')

plt.subplot(1, 2, 2)
for i in range(5):
    plt.plot(ln_paths['time_points'], ln_paths['vol_paths'][:, i])
plt.title('Log-Normal SV - Volatility Paths')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.tight_layout()
plt.show()

# Heston SV model
kappa_h = 2.0    # Mean reversion speed
theta_h = 0.04   # Long-term variance
sigma_h = 0.3    # Volatility of variance
rho_h = -0.7     # Correlation
v0_h = 0.04      # Initial variance

heston_model = HestonSV(kappa_h, theta_h, sigma_h, rho_h)

# Price European options
heston_call_price = heston_model.price_european_call(S, K, v0_h, T, r)
heston_put_price = heston_model.price_european_put(S, K, v0_h, T, r)

print(f"Heston SV Call Price: {heston_call_price:.6f}")  # 10.394219
print(f"Heston SV Put Price: {heston_put_price:.6f}")    # 5.517161
```

## License

MIT

## Credits

Based on the option pricing implementation by Artur Sepp.
