import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def buildTree(S, vol, T, N):
    dt = T / N

    matrix = np.zeros((N + 1, N + 1))

    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    for i in range(N + 1):
        for j in range(i + 1):
            matrix[i, j] = S * (u ** (j)) * (d ** (i-j))

    return matrix

def valueOptionMatrix(tree, T, r, K, vol, N):
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
        matrix[rows - 1, c] = max(S - K, 0)


    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = matrix[i + 1, j]
            up = matrix[i + 1, j + 1]
            matrix[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

    return matrix

def valueOptionMatrixAmerican(tree, T, r, K, vol, N):
    dt = T / N

    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u

    p = (np.exp(r * dt) - d) / (u - d)

    columns = tree.shape[1]
    rows = tree.shape[0]

    price_tree = np.zeros((rows, columns))

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        price_tree[rows - 1, c] = max(S - K, 0)


    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = price_tree[i + 1, j]
            up = price_tree[i + 1, j + 1]
            price_tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

            S = tree[i, j]
            price_tree[i, j] = max(price_tree[i, j], S - K)

    return price_tree

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
        error_list.append(np.abs(price[0, 0] - priceAnalytical))

    plt.plot(sigma_list, tree_list, label="Binomial tree")
    plt.plot(sigma_list, analytical_list, label="Black Scholes")
    plt.xlabel("Volatility")
    plt.ylabel("Option price")
    plt.legend()
    plt.savefig("volatility.png")
    plt.clf()

    plt.plot(sigma_list, error_list)
    plt.xlabel("Volatility")
    plt.ylabel("Error")
    plt.savefig("volatility_error.png")
    plt.clf()

def part_3():
    sigma = 0.2
    diff_list = []

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    for N in np.arange(1, 100):
        tree = buildTree(S, sigma, T, N)
        price = valueOptionMatrix(tree, T, r, K, sigma, N)

        diff = np.abs(price[0, 0] - priceAnalytical)
        diff_list.append(diff)

    plt.plot(np.arange(1, 100), diff_list)
    plt.xlabel("Number of steps in binomial tree")
    plt.ylabel("Error")
    plt.savefig("diff.png")
    plt.clf()

def part_4():
    delta_list_tree = []
    delta_list_analytical = []

    for sigma in sigma_list:
        tree = buildTree(S, sigma, T, N)
        price = valueOptionMatrix(tree, T, r, K, sigma, N)

        d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        delta_tree = (price[1, 0] - price[1, 1]) / (tree[1, 0] - tree[1, 1])
        delta_analytical = norm.cdf(d1)
        
        delta_list_tree.append(delta_tree)
        delta_list_analytical.append(delta_analytical)

    plt.plot(sigma_list, delta_list_tree, label="Binomial tree")
    plt.plot(sigma_list, delta_list_analytical, label="Black Scholes")
    plt.xlabel("Volatility")
    plt.ylabel("Delta")
    plt.legend()
    plt.savefig("delta.png")
    plt.clf()

def part_5():
    european_list = []
    american_list = []
    diff_list = []

    for sigma in sigma_list:
        tree = buildTree(S, sigma, T, N)
        price_european = valueOptionMatrix(tree, T, r, K, sigma, N)

        tree = buildTree(S, sigma, T, N)
        price_american = valueOptionMatrixAmerican(tree, T, r, K, sigma, N)

        european_list.append(price_european[0, 0])
        american_list.append(price_american[0, 0])

        diff_list.append(price_american[0, 0] - price_european[0, 0])

    plt.plot(sigma_list, european_list, label="European")
    plt.plot(sigma_list, american_list, label="American")
    plt.xlabel("Volatility")
    plt.ylabel("Option price")
    plt.legend()
    plt.savefig("american_european.png")
    plt.clf()

    plt.plot(sigma_list, diff_list)
    plt.xlabel("Volatility")
    plt.ylabel("Difference")
    plt.savefig("american_european_diff.png")
    plt.clf()


# part_1()
# part_2()
# part_3()
# part_4()
part_5()

sigma = 0.1
tree = buildTree(S, sigma, T, N)
price_american = valueOptionMatrixAmerican(tree, T, r, K, sigma, N)
price_european = valueOptionMatrix(tree, T, r, K, sigma, N)

print(price_european[1])
print(price_american[1])