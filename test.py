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



def ass1():
    sigma = 0.20
    S = 100
    T = 1

    K = 99
    r = 0.06

    for N in range(1, 50):
        tree = buildTree(S, sigma, T, N)
        matrix = valueOptionMatrix(tree, T, r, K, sigma, N)
        print(matrix[0][0])

def ass2():
    sigma = 0.20
    S = 100
    T = 1

    K = 99
    r = 0.06

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return priceAnalytical

print(ass1())
print(ass2())