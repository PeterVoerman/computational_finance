import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib

matplotlib.rcParams.update({'font.size': 16})


def monte_carlo_asian_arithmetic(S0, K, T, r, sigma, N, M):
    delta_t = T / N
    S = np.full((M), S0, dtype='float64')
    total_S = np.zeros_like(S)

    for i in range(N):
        S += S * (r * delta_t + sigma * np.sqrt(delta_t) * np.random.normal(0, 1, M))
        total_S += S

    avg_S = total_S / N

    payoff = np.maximum(avg_S - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)

    return price, np.exp(-r * T) * payoff

def monte_carlo_asian_geometric(S0, K, T, r, sigma, N, M):
    delta_t = T / N
    S = np.full((M), S0, dtype='float64')
    total_S = np.ones_like(S)

    for i in range(N):
        S += S * (r * delta_t + sigma * np.sqrt(delta_t) * np.random.normal(0, 1, M))
        total_S *= S ** (1 / N)

    avg_S = total_S

    payoff = np.maximum(avg_S - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)

    return price
        
S0 = 100
K = 99
T = 1
r = 0.06
sigma = 0.2
N = 260
M = int(1e4)

sigma_tilde = sigma * np.sqrt((2 * N + 1) / (6 * (N + 1))) 
r_tilde = ((r - sigma ** 2 / 2) + sigma_tilde ** 2) / 2

d1_tilde = (np.log(S0 / K) + (r_tilde + sigma_tilde ** 2 / 2) * T) / (sigma_tilde * np.sqrt(T))
d2_tilde = (np.log(S0 / K) + (r_tilde - sigma_tilde ** 2 / 2) * T) / (sigma_tilde * np.sqrt(T))

geometric_analytical = np.exp(-r * T) * (S0 * np.exp(r_tilde * T) * norm.cdf(d1_tilde) - K * norm.cdf(d2_tilde))
error_list = []
std_list =  []

for M in np.logspace(0, 4, 9):
    print(f"M: {M}")
    geometric_list = []

    for j in range(1000):
        geometric = monte_carlo_asian_geometric(S0, K, T, r, sigma, N, int(M))
        geometric_list.append(geometric)

    error = np.abs(np.mean(geometric_list) - geometric_analytical)
    error_list.append(error)
    std_list.append(np.std(geometric_list))

plt.plot(np.logspace(0, 4, 9), error_list, label="Error")
plt.xlabel("M")
plt.ylabel("Error")
plt.xscale("log")
plt.tight_layout()
plt.savefig("asian_error.png")
plt.clf()

plt.plot(np.logspace(0, 4, 9), std_list, label="Std")
plt.xlabel("M")
plt.ylabel("$\sigma$")
plt.xscale("log")
plt.tight_layout()
plt.savefig("asian_std.png")
plt.clf()

monte_carlo_list = []
control_variate_list = []

M = int(1e3)


monte_carlo_list = []
control_variate_list = []

for i in range(100):
    print(f"{i/100 * 100:.2f}%", end="\r")

    monte_carlo = monte_carlo_asian_arithmetic(S0, K, T, r, sigma, N, M)
    monte_carlo_list.append(monte_carlo[0])

    geometric_monte_carlo = monte_carlo_asian_geometric(S0, K, T, r, sigma, N, M)

    # print(monte_carlo[1])

    control_variate = monte_carlo[0] - (geometric_monte_carlo[0] - geometric_analytical) * (np.std(monte_carlo[1]) / np.std(geometric_monte_carlo[1])) * np.corrcoef(monte_carlo[1], geometric_monte_carlo[1])[0, 1]
    control_variate_list.append(control_variate)
    # print((np.std(monte_carlo[1]) / np.std(geometric_monte_carlo[1])) * np.corrcoef(monte_carlo[1], geometric_monte_carlo[1])[0, 1])

monte_carlo_list = np.array(monte_carlo_list)
control_variate_list = np.array(control_variate_list)

print(f"Monte Carlo mean: {np.mean(monte_carlo_list)}")
print(f"Monte Carlo std: {np.std(monte_carlo_list)}")

print(f"Control variate mean: {np.mean(control_variate_list)}")
print(f"Control variate std: {np.std(control_variate_list)}")

plt.hist(monte_carlo_list, bins=100, alpha=0.5, label="Monte Carlo")
plt.hist(control_variate_list, bins=100, alpha=0.5, label="Control Variate")
plt.legend()
plt.show()
