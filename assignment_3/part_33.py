from math import log, pi, sin, cos
from numpy import exp
import numpy as np
from scipy.stats import norm

S0 = 110
K = 110
N = 64
sigma = 0.3

T = 1
r = 0.04

t = 1

a = log(S0 / K) + r * T - 12 * (sigma ** 2 * T) ** (1 / 2)
b = log(S0 / K) + r * T + 12 * (sigma ** 2 * T) ** (1 / 2)

# b - a, vaak nodig
bma = 24 * (sigma ** 2 * T) ** (1 / 2)

chi = lambda k: (1 + ((k * pi) / bma) ** 2) ** (-1) * (((-1) ** k) * exp(b) - cos(k * pi * a / bma) + ((k * pi) / bma) * sin(k * pi * a / bma))
psi = lambda k: b if k == 0 else (bma / (k * pi)) * sin(k * pi * a / bma)

Vk = lambda k: (2 / bma) * K * (chi(k) - psi(k))

phi = lambda u: exp(-0.5 * sigma ** 2 * T * u ** 2 + (u * (r - 0.5 * sigma ** 2)*T) * 1j)
Fk = lambda k: 2/bma * (phi((k * pi / bma)) * exp(-k * a *pi / bma * 1j)).real

V_xt = 0.5 * Fk(0) * Vk(0)

for k1 in range(1, N - 1):
    V_xt += Fk(k1) * Vk(k1)
V_xt *= exp(-r * t) * bma / 2

d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
priceAnalytical = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
print(f"Black scholes: {priceAnalytical}")
print(f"COS: {V_xt}")