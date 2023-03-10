import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 16})

start = time.time()

def likelihood_ratio():

    T = 1
    M = 1000

    K = 99
    S0_start = 100
    sigma = 0.2
    r = 0.06

    M = int(1e5)

    S0 = S0_start
    delta_list = []

    for j in range(5000):
        print(f"{j/5000*100:.2f}%", end="\r")

        random_numbers = np.random.normal(0, 1, M)

        S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * random_numbers)
        
        payoff_list = [1 if S[i] > K else 0 for i in range(M)]
        payoff_list = np.array(payoff_list)

        delta = np.mean(np.exp(-r * T) * payoff_list * random_numbers / (sigma * S0 * np.sqrt(T)))
        delta_list.append(delta)

    print(f"Average delta: {np.mean(delta_list)}")
    print(f"Standard deviation: {np.std(delta_list)}")

    return delta_list

def bump_and_revalue(epsilon):
    T = 1
    M = 1000

    K = 99
    S0_start = 100
    sigma = 0.2
    r = 0.06

    M = int(1e5)

    S0 = S0_start
        
    value_list = []
    delta_list = []

    for j in range(5000):
        print(f"{j/5000*100:.2f}%", end="\r")
        for S0 in [S0_start, S0_start + epsilon]:
            np.random.seed(j) # Comment to use different seeds for the bumped and unbumped estimates

            S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  
            payoff_sum = sum(S > K)
            option_value = np.exp(-r * T) * payoff_sum / M
            value_list.append(option_value)

        delta = (value_list[-1] - value_list[-2]) / epsilon
        delta_list.append(delta)

    print(f"Average delta: {np.mean(delta_list)}")
    print(f"Standard deviation: {np.std(delta_list)}")

    return delta_list

def bump_and_revalue_test_epsilon(best_delta):

    T = 1
    M = 1000

    K = 99
    S0_start = 100
    sigma = 0.2
    r = 0.06

    M = int(1e4)

    S0 = S0_start

    avg_delta_list = []

    min_epsilon = -4
    max_epsilon = 1
    steps = 2 * (max_epsilon - min_epsilon) + 1

    for epsilon in np.logspace(min_epsilon, max_epsilon, steps):
        print(f"epsilon = {epsilon}")   
        
        value_list = []
        delta_list = []

        for j in range(5000):
            print(f"{j/5000*100:.2f}%", end="\r")
            for S0 in [S0_start, S0_start + epsilon]:
                np.random.seed(j) # Comment to use different seeds for the bumped and unbumped estimates

                S = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * T ** 0.5 * np.random.normal(0, 1, M))  
                payoff_sum = sum(S > K)
                option_value = np.exp(-r * T) * payoff_sum / M
                value_list.append(option_value)

            delta = (value_list[-1] - value_list[-2]) / epsilon
            delta_list.append(delta)

        print(f"Average delta: {np.mean(delta_list)}")
        print(f"Standard deviation: {np.std(delta_list)}")
        avg_delta_list.append(np.mean(delta_list))

    plt.plot(np.logspace(min_epsilon, max_epsilon, steps), avg_delta_list)
    plt.axhline(y=best_delta, color="red", label="Likelihood ratio delta", linestyle="--")
    plt.legend()
    plt.xscale("log")
    plt.savefig("digital_epsilon.png")
    plt.clf()


likelihood = likelihood_ratio()
bump_revalue = bump_and_revalue(0.05)
bump_and_revalue_test_epsilon(np.mean(likelihood))

print(len(likelihood))
print(len(bump_revalue))

plt.hist(likelihood, bins=100, label="Likelihood ratio", alpha=0.5)
plt.hist(bump_revalue, bins=100, label="Bump and revalue", alpha=0.5)
plt.xlabel("$\delta$")
plt.legend()
plt.savefig("digital_hist.png")

end = time.time()
print(f"Time elapsed: {end - start}")