import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

start = time.time()
T = 1
M = 1000

K = 99
S0 = 100
sigma = 0.2
r = 0.06

def vary_M():
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    error_list = []
    stdev_list = []

    M_min = 0
    M_max = 6
    steps = 2 * (M_max - M_min) + 1

    for M in np.logspace(M_min, M_max, steps, dtype=int):
        print(f"M = {M}")
        value_list = []
        for j in range(1000):
            print(f"{j/1000*100:.2f}%", end="\r")

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  

            payoff_sum = sum(np.maximum(K - S, 0))
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        print(f"Average price: {np.mean(value_list)}")
        print(f"Standard deviation: {np.std(value_list)}")

        error_list.append(abs(np.mean(value_list) - priceAnalytical))
        stdev_list.append(np.std(value_list))

    plt.plot(np.logspace(M_min, M_max, steps, dtype=int), error_list, label="Error")
    plt.xscale("log")
    plt.xlabel("M")
    plt.ylabel("Error")
    plt.savefig("error_varying_M.png")
    plt.clf()

    plt.plot(np.logspace(M_min, M_max, steps, dtype=int), stdev_list, label="Standard deviation")
    plt.xscale("log")
    plt.xlabel("M")
    plt.ylabel(f'$\sigma$')
    plt.savefig("stdev_varying_M.png")
    plt.clf()

def vary_K():
    M = int(1e5)

    K_start = 80
    K_end = 120
    K_step_size = 5

    error_list = []
    stdev_list = []
    price_list = []

    for K in range(K_start, K_end + 1, K_step_size):
        print(f"K = {K}")

        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        priceAnalytical = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        value_list = []
        for j in range(1000):
            print(f"{j/1000*100:.2f}%", end="\r")

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  

            payoff_sum = sum(np.maximum(K - S, 0))
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        print(f"Average price: {np.mean(value_list)}")
        print(f"Standard deviation: {np.std(value_list)}")

        price_list.append(np.mean(value_list))
        error_list.append((abs(np.mean(value_list) - priceAnalytical)))
        stdev_list.append(np.std(value_list)/np.mean(value_list))

    plt.plot(range(K_start, K_end + 1, K_step_size), price_list, label="Price")
    plt.xlabel("K")
    plt.ylabel("Price")
    plt.savefig("price_varying_K.png")
    plt.clf()

    plt.plot(range(K_start, K_end + 1, K_step_size), error_list, label="Error")
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.savefig("error_varying_K.png")
    plt.clf()

    plt.plot(range(K_start, K_end + 1, K_step_size), stdev_list, label="Standard deviation")
    plt.xlabel("K")
    plt.ylabel(f'$\sigma$')
    plt.savefig("stdev_varying_K.png")
    plt.clf()


def vary_sigma():
    M = int(1e5)

    sigma_start = 0
    sigma_end = 1
    sigma_step_size = 0.2

    error_list = []
    stdev_list = []
    price_list = []

    for sigma in np.arange(sigma_start, sigma_end + sigma_step_size, sigma_step_size):
        print(f"sigma = {sigma}")

        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        priceAnalytical = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        value_list = []
        for j in range(1000):
            print(f"{j/1000*100:.2f}%", end="\r")

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  

            payoff_sum = sum(np.maximum(K - S, 0))
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        print(f"Average price: {np.mean(value_list)}")
        print(f"Standard deviation: {np.std(value_list)}")

        price_list.append(np.mean(value_list))
        error_list.append((abs(np.mean(value_list) - priceAnalytical)))
        stdev_list.append(np.std(value_list)/np.mean(value_list))

    plt.plot(np.arange(sigma_start, sigma_end + sigma_step_size, sigma_step_size), price_list, label="Price")
    plt.xlabel("Volatility")
    plt.ylabel("Price")
    plt.savefig("price_varying_sigma.png")
    plt.clf()

    plt.plot(np.arange(sigma_start, sigma_end + sigma_step_size, sigma_step_size), error_list, label="Error")
    plt.xlabel("Volatility")
    plt.ylabel("Error")
    plt.savefig("error_varying_sigma.png")
    plt.clf()

    plt.plot(np.arange(sigma_start, sigma_end + sigma_step_size, sigma_step_size), stdev_list, label="Standard deviation")
    plt.xlabel("Volatility")
    plt.ylabel("Standard deviation")
    plt.savefig("stdev_varying_sigma.png")
    plt.clf()


vary_M()
vary_K()
vary_sigma()


end = time.time()
print(f"Time elapsed: {end - start}")