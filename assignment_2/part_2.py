import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

start = time.time()

T = 1
M = 1000

K = 99
S0_start = 100
sigma = 0.2
r = 0.06

M = 10000

S0 = S0_start
d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
priceAnalyticalCall = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
delta_analytical = norm.cdf(d1)

avg_delta_list = []

min_epsilon = -10
max_epsilon = -2
steps = 2 * (max_epsilon - min_epsilon) + 1

for epsilon in np.logspace(min_epsilon, max_epsilon, steps):
    print(f"epsilon = {epsilon}")   
    
    value_list = []
    delta_list = []

    for j in range(1000):
        print(f"{j/1000*100:.2f}%", end="\r")
        for S0 in [S0_start, S0_start + epsilon]:
            np.random.seed(j) # Comment to use different seeds for the bumped and unbumped estimates

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  
            payoff_sum = sum(np.maximum(S - K, 0))
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        delta = (value_list[-1] - value_list[-2]) / epsilon
        delta_list.append(delta)

    print(f"Average delta: {np.mean(delta_list)}")
    print(f"Standard deviation: {np.std(delta_list)}")
    avg_delta_list.append(np.mean(delta_list))

plt.plot(np.logspace(min_epsilon, max_epsilon, steps), avg_delta_list, label="Bump and revalue delta")
plt.hlines(delta_analytical, 10**min_epsilon, 10**max_epsilon, label="Analytical delta", colors="red", linestyles="dashed")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.xlabel(f"$\epsilon$")
plt.ylabel("$\delta$")
plt.tight_layout()
plt.savefig("same_seed.png")

end = time.time()
print(f"Time elapsed: {end - start}")