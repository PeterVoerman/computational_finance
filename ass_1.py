import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats


plt.rcParams.update({'font.size': 15})

def buildTree(S, vol, T, N):
    dt = T / N

    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    for i in range(N + 1):
        for j in range(i + 1):
            matrix[i, j] = S * (u ** (j)) * (d ** (i-j))

    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N, call=True):
    dt = T / N

    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    p = (np.exp(r * dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]

    # create a new matrix to prevent problems with global variables
    matrix = np.zeros_like(tree)

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        if call:
            matrix[rows - 1, c] = max(S - K, 0)
        else:
            matrix[rows - 1, c] = max(K - S, 0)


    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = matrix[i + 1, j]
            up = matrix[i + 1, j + 1]
            matrix[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

    return matrix

def valueOptionMatrixAmerican(tree, T, r, K, vol, N, call=True):
    dt = T / N

    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    p = (np.exp(r * dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]

    price_tree = np.zeros((rows, columns))

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        if call:
            price_tree[rows - 1, c] = max(S - K, 0)
        else:
            price_tree[rows - 1, c] = max(K - S, 0)


    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = price_tree[i + 1, j]
            up = price_tree[i + 1, j + 1]
            price_tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

            S = tree[i, j]
            if call:
                price_tree[i, j] = max(price_tree[i, j], S - K)
            else:
                price_tree[i, j] = max(price_tree[i, j], K - S)

    return price_tree


def approxGbmEuler(M, T, r, vol, S0):
    delta_t = T/M
    price = [S0]

    for m in np.arange(M):
        Z_m = np.random.normal(0,1)
        price.append(price[-1] + r*price[-1]*delta_t + vol * price[-1] * np.sqrt(delta_t) * Z_m)
    return price

# T = 1
# M = 100
# price = approxGbmEuler(M, T, 0.05, 0.2, 50)
# plt.plot(np.arange(0,T+T/M,T/M),price)
# plt.show()
# quit()

def example():
    sigma = 0.25
    S = 80
    T = 1
    N = 2

    K = 85
    r = 0.1

    N = np.arange(1, 50)

    analytical_list = []
    approximated_list = []

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    analytical_list = [priceAnalytical] * len(N)

    for n in N:
        print(n, end='\r')
        treeN = buildTree(S, sigma, T, n)
        priceApproximatedly = valueOptionMatrix(treeN, T, r, K, sigma, n)

        approximated_list.append(priceApproximatedly[0, 0])

    error_list = np.abs(np.array(analytical_list) - np.array(approximated_list))

    plt.plot(N, analytical_list, label="Analytical")
    plt.plot(N, approximated_list, label="Approximated")
    plt.xlabel("Number of steps in binomial tree")
    plt.ylabel("Option price")
    plt.legend()
    plt.savefig("comparison.png")
    plt.clf()

    plt.plot(N, error_list)
    plt.xlabel("Number of steps in binomial tree")
    plt.ylabel("Error")
    plt.savefig("error.png")
    plt.clf()

sigma = 0.2
S = 100
K = 99
T = 1
r = 0.06
N = 50

num_steps = 100
sigma_list = np.linspace(1 / num_steps, 1, num_steps)

def part_1():
    print("1.1")

    tree = buildTree(S, sigma, T, N)
    price = valueOptionMatrix(tree, T, r, K, sigma, N)

    print(f"Binomial tree: {price[0, 0]:.4f}")

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    print(f"Black Scholes: {priceAnalytical:.4f}")

def part_2():
    tree_list = []
    analytical_list = []
    error_list = []

    for sigma in sigma_list:
        tree = buildTree(S, sigma, T, N)
        price = valueOptionMatrix(tree, T, r, K, sigma, N)

        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        tree_list.append(price[0, 0])
        analytical_list.append(priceAnalytical)
        error_list.append((price[0, 0] - priceAnalytical)/priceAnalytical)

    plt.plot(sigma_list, tree_list, label="Binomial tree")
    plt.plot(sigma_list, analytical_list, label="Black Scholes")
    plt.xlabel("Volatility")
    plt.ylabel("Option price ($)")
    plt.xlim(0, 1)
    plt.ylim(0, 45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("volatility.png")
    plt.clf()

    plt.plot(sigma_list, error_list)
    plt.xlabel("Volatility")
    plt.ylabel("Relative error")
    plt.xlim(0, 1)
    plt.ylim(-0.15, 0.05)
    # plt.yticks(np.arange(0, 0.16, 0.03))
    plt.tight_layout()
    plt.savefig("volatility_error.png")
    plt.clf()

def part_3():
    sigma = 0.2
    diff_list = []

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    for N in np.arange(1, 101):
        tree = buildTree(S, sigma, T, N)
        price = valueOptionMatrix(tree, T, r, K, sigma, N)

        diff = np.abs(price[0, 0] - priceAnalytical)
        diff_list.append(diff)
    plt.plot(np.arange(1, 101), diff_list)
    plt.xlabel("Number of steps in binomial tree")
    plt.ylabel("Error ($)")
    plt.xlim(0, 100)
    plt.ylim(0, 1.75)
    plt.tight_layout()
    plt.savefig("diff.png")
    plt.show()
    plt.clf()

def part_4():
    delta_list_tree = []
    delta_list_analytical = []

    for sigma in sigma_list:
        tree = buildTree(S, sigma, T, N)
        price = valueOptionMatrix(tree, T, r, K, sigma, N)

        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

        delta_tree = (price[1, 0] - price[1, 1]) / (tree[1, 0] - tree[1, 1])
        delta_analytical = norm.cdf(d1)
        
        delta_list_tree.append(delta_tree)
        delta_list_analytical.append(delta_analytical)

    plt.plot(sigma_list, delta_list_tree, label="Binomial tree", color="red")
    plt.plot(sigma_list, delta_list_analytical, label="Black Scholes", linestyle="--", color="blue")
    plt.xlabel("Volatility")
    plt.ylabel("Delta")
    plt.xlim(0, 1)
    plt.ylim(0.6, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("delta.png")
    plt.clf()

def part_5():
    for call in [True, False]:
        european_list = []
        american_list = []
        diff_list = []

        for sigma in sigma_list:
            tree = buildTree(S, sigma, T, N)
            price_european = valueOptionMatrix(tree, T, r, K, sigma, N, call)

            tree = buildTree(S, sigma, T, N)
            price_american = valueOptionMatrixAmerican(tree, T, r, K, sigma, N, call)

            european_list.append(price_european[0, 0])
            american_list.append(price_american[0, 0])

            diff_list.append(price_american[0, 0] - price_european[0, 0])

        if call:
            plt.plot(sigma_list, european_list, label="European", color="blue")
            plt.plot(sigma_list, american_list, label="American", linestyle="--", color="red")
            plt.xlabel("Volatility")
            plt.ylabel("Option pric ($)")
            plt.xlim(0, 1)
            plt.ylim(0, 45)
            plt.legend()
            plt.tight_layout()
            plt.savefig("american_european_call.png")
            plt.clf()

            plt.plot(sigma_list, diff_list)
            plt.xlabel("Volatility")
            plt.ylabel("Difference (American - European) ($)")
            plt.xlim(0, 1)
            plt.ylim(-1, 1)
            plt.tight_layout()
            plt.savefig("american_european_diff_call.png")
            plt.clf()
        else:
            plt.plot(sigma_list, european_list, label="European", color="blue")
            plt.plot(sigma_list, american_list, label="American", color="red")
            plt.xlabel("Volatility")
            plt.ylabel("Option price ($)")
            plt.xlim(0, 1)
            plt.ylim(0, 35)
            plt.legend()
            plt.tight_layout()
            plt.savefig("american_european_put.png")
            plt.clf()

            plt.plot(sigma_list, diff_list)
            plt.xlabel("Volatility")
            plt.ylabel("Difference (American - European) ($)")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig("american_european_diff_put.png")
            plt.clf()

def part33():
    r = 0.06
    starting_price = 100
    K = 99
    T = 1
    time_steps = 365

    # same volatility experiment
    adjust_freq = 7
    vol_euler = 0.2
    vol_bs = 0.2
    iterations = 1000
    ############################################################################
    money_list = []
    for i in range(iterations):
        money = 0
        price = approxGbmEuler(time_steps, T, r, vol_euler, starting_price)
        time = []
        for x in np.arange(0,T+T/time_steps,T/time_steps):
            if x <= 1:
                time.append(x)
    
        # plt.plot(time,price)
        # plt.show()

        hedge_delta_t = 0

        for t in np.arange(0, time_steps, adjust_freq):
            previous_delta = hedge_delta_t

            # calculate delta
            d1 = (np.log(price[t] / K) + (r + vol_bs ** 2 / 2) * (T-t/time_steps)) / (vol_bs * np.sqrt(T-t/time_steps))

            hedge_delta_t = norm.cdf(d1)
            # print(hedge_delta_t)

            # adjust hedge position, calculate net money
            money -= price[t] * (hedge_delta_t - previous_delta)
            # if t % 10 == 0:
            #     print(f'price {price[t]}')
            #     print(f'hedge_delta {hedge_delta_t}')
            #     print(f'money {money}')
            #     print()
        
        # strike day
        if price[-1] > K:
            option_price = price[-1] - K
        else:
            option_price = 0

        # print('Final')
        # print(f'stock price {price[-1]}')
        # print(f'money {money}')
        # print(f'hedge_delta {hedge_delta_t}')
        # print(f'option price {option_price}')

        money_at_strike = money + hedge_delta_t * price[-1] - option_price
        # print(f'money at strike {money_at_strike}')

        money_list.append(money_at_strike)

    error = np.std(money_list) * 1.96

    
    print(f'Money after: {sum(money_list)/len(money_list):.2f} +- {error:.2f}')

    # different volatility experiment


    return


# part_1()
# part_2()
part_3()
# part_4()
# part_5()
# part33()

# sigma = 0.2
# tree = buildTree(S, sigma, T, N)
# price_american = valueOptionMatrixAmerican(tree, T, r, K, sigma, N, False)
# tree = buildTree(S, sigma, T, N)
# price_european = valueOptionMatrix(tree, T, r, K, sigma, N, False)

# print(price_american[0, 0])
# print(price_european[0, 0])
