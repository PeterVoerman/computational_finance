import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

start = time.time()
T = 1
N = 100
M = 1000

K = 99
S0_start = 100
sigma = 0.2
r = 0.06
epsilon = 0.01

delta_t = T / N

payoff_sum = 0

M = 10000
for epsilon in np.logspace(-10, 0, 11):
# for epsilon in [1, 10]:
    print(f"epsilon = {epsilon}")   
    
    value_list = []
    delta_list = []

    for j in range(1000):
        print(f"{j/1000*100:.2f}%", end="\r")
        for S0 in [S0_start, S0_start + epsilon]:
            np.random.seed(j)
            
            payoff_sum = 0

            S = np.array([S0] * M, dtype=np.float64)
            
            for k in range(N):
                S += S * (r * delta_t + sigma * np.random.normal(0, 1, M) * np.sqrt(delta_t))

            payoff_sum = sum(np.maximum(S - K, 0))

            option_value = np.exp(-r * T) * payoff_sum / M

            value_list.append(option_value)
        delta = (value_list[-1] - value_list[-2]) / epsilon
        delta_list.append(delta)

    # print(value_list)
    # print(delta_list)
    print(f"Average delta: {np.mean(delta_list)}")
    print(f"Standard deviation: {np.std(delta_list)}")

end = time.time()
print(f"Time elapsed: {end - start}")