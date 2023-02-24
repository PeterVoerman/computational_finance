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

    for c in np.arange(columns):
        S = tree[rows - 1, c]
        tree[rows - 1, c] = max(S - K, 0)


    for i in np.arange(rows - 1)[::-1]:
        for j in np.arange(i + 1):
            down = tree[i + 1, j]
            up = tree[i + 1, j + 1]
            tree[i, j] = np.exp(-r * dt) * (p * up + (1 - p) * down)

    return tree

def example():
    sigma = 0.25
    S = 80
    T = 1
    N = 2

    K = 85
    r = 0.1

    N = np.arange(1, 300)

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
example()

# tree = buildTree(S, sigma, T, N)
# print(tree)
# matrix = valueOptionMatrix(tree, T, r, K, sigma, N)

# print(matrix)