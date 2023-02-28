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

for M in [10, 100, 1000, 10000, 100000]:
    print(f"M = {M}")
    value_list = []
    for j in range(1000):
        print(f"{j/1000*100:.2f}%", end="\r")
        payoff_sum = 0
        for i in range(M):
            # print(f"{i/M*100:.2f}%", end="\r")
            S = S0
            for j in range(N):
                S += S * (r * delta_t + sigma * np.random.normal(0, 1) * np.sqrt(delta_t))

            payoff_sum += max(S - K, 0)

        option_value = np.exp(-r * T) * payoff_sum / M

        value_list.append(option_value)

    print(f"Average price: {np.mean(value_list)}")
    print(f"Standard deviation: {np.std(value_list)}")