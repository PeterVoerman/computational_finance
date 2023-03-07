import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



def monte_carlo_asian(S0, K, T, r, sigma, N, M):
    delta_t = T / N
    S = np.full((M), S0, dtype='float64')
    total_S = np.zeros_like(S)

    for i in range(N):
        print(f"{i/N * 100:.2f}%")

        S += S * (r * delta_t + sigma * np.sqrt(delta_t) * np.random.normal(0, 1, M))
        total_S += S

    avg_S = total_S / N

    payoff = np.maximum(avg_S - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)

    return price
        
S0 = 100
K = 99
T = 1
r = 0.06
sigma = 0.2
N = 260
M = int(1e7)

print(f"Monte Carlo price: {monte_carlo_asian(S0, K, T, r, sigma, N, M)}")

sigma_tilde = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1))) 
r_tilde = ((r - sigma ** 2 / 2) + sigma_tilde ** 2) / 2

d1_tilde = (np.log(S0 / K) + (r_tilde + sigma_tilde ** 2 / 2) * T) / (sigma_tilde * np.sqrt(T))
d2_tilde = (np.log(S0 / K) + (r_tilde - sigma_tilde ** 2 / 2) * T) / (sigma_tilde * np.sqrt(T))

price_analytical = np.exp(-r * T) * (S0 * np.exp(r_tilde * T) * norm.cdf(d1_tilde) - K * norm.cdf(d2_tilde))

print(f"Analytical price: {price_analytical}")