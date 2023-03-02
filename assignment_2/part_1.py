import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

start = time.time()
T = 1
N = 100
M = 1000

K = 99
S0 = 100
sigma = 0.2
r = 0.06

delta_t = T / N

payoff_sum = 0

d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
delta_analytical = norm.cdf(d1)
print(delta_analytical)

quit()

for M in np.logspace(0, 6, 7, dtype=int):
    print(f"M = {M}")
    value_list = []
    for j in range(1000):
        print(f"{j/1000*100:.2f}%", end="\r")
        payoff_sum = 0

        S = np.array([S0] * M, dtype=np.float64)
        
        for j in range(N):
            S += S * (r * delta_t + sigma * np.random.normal(0, 1, M) * np.sqrt(delta_t))

        payoff_sum = sum(np.maximum(S - K, 0))

        option_value = np.exp(-r * T) * payoff_sum / M

        value_list.append(option_value)

    print(f"Average price: {np.mean(value_list)}")
    print(f"Standard deviation: {np.std(value_list)}")

end = time.time()
print(f"Time elapsed: {end - start}")