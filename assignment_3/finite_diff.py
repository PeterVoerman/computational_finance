import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import norm

def finite_diff_ftcs(S0, K, r, sigma, T, dx, dt, M1, M2):
    Nt = int(T / dt)
    Nx = int((M2 + M1) / dx) + 2

    A_diag_1 = (-(r - 0.5 * sigma ** 2) * dt / (2 * dx) + 0.5 * sigma ** 2 * dt / dx ** 2) * np.ones(Nx - 1)
    A_diag_2 = (1 - sigma ** 2 * dt / dx ** 2 - r * dt) * np.ones(Nx)
    # A_diag_2 = (1 - 2 * (r - 0.5 * sigma ** 2) * (dt) / (2 * dx) - 0.5 * (sigma ** 2) * (dt) / (dx ** 2) - r * dt) * np.ones(Nx)
    A_diag_3 = ((r - 0.5 * sigma ** 2) * dt / (2 * dx) + 0.5 * sigma ** 2 * dt / dx ** 2) * np.ones(Nx - 1)

    A_diag_2[0] = 0
    A_diag_3[0] = 0

    A_diag_1[-1] = 0
    A_diag_2[-1] = 0

    offset = np.zeros(Nx)
    offset[-1] = np.exp(M2)

    A = sparse.diags([A_diag_1, A_diag_2, A_diag_3], [-1, 0, 1], format='csc')

    # print(A.toarray())

    S = np.exp(np.linspace(-M1, M2, Nx))
    V = np.maximum(S - K, 0)

    # V = np.full(Nx, max(S0 - K, 0))

    print(V.shape)

    for n in range(Nt):
        V = A @ V + offset

    print(list(V))
    # print(np.linspace(-M1, M2, d))
    
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    priceAnalytical = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    plt.plot(S, V, label="numeric")
    # plt.plot(S, priceAnalytical, label="bs")
    plt.legend()
    plt.show()

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
dx = 0.01
dt = 0.01
M1 = 10
M2 = 10

finite_diff_ftcs(S0, K, r, sigma, T, dx, dt, M1, M2)