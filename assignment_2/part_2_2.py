import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

start = time.time()
T = 1
M = 1000

K = 99
S0_start = 100
sigma = 0.2
r = 0.06

M = 10000

for epsilon in np.logspace(-5, 2, 8):
    print(f"epsilon = {epsilon}")   
    
    value_list = []
    delta_list = []

    for j in range(1000):
        print(f"{j/1000*100:.2f}%", end="\r")
        for S0 in [S0_start, S0_start + epsilon]:
            np.random.seed(j)

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  
            payoff_sum = sum(S > K)
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        delta = (value_list[-1] - value_list[-2]) / epsilon
        delta_list.append(delta)

    print(f"Average delta: {np.mean(delta_list)}")
    print(f"Standard deviation: {np.std(delta_list)}")

end = time.time()
print(f"Time elapsed: {end - start}")