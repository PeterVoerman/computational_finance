import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def finite_diff(S0, K, r, sigma, T, dx, dt, M1, M2):
    N = int(T / dt)

    A_diag_1 = ((r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(N - 1)
    A_diag_2 = (1 + 0.5 * sigma ** 2 * dt / dx ** 2 + 0.5 * r * dt) * np.ones(N)
    A_diag_3 = (-(r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(N - 1)

    B_diag_1 = (-(r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(N - 1)
    B_diag_2 = (1 - 0.5 * sigma ** 2 * dt / dx ** 2 - 0.5 * r * dt) * np.ones(N)
    B_diag_3 = ((r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(N - 1)

    # A_diag_2[0] = 0
    # A_diag_3[0] = 0
    # A_diag_1[-1] = 2
    # A_diag_2[-1] = 2

    A = sparse.diags([A_diag_1, A_diag_2, A_diag_3], [-1, 0, 1], shape=(N, N))
    B = sparse.diags([B_diag_1, B_diag_2, B_diag_3], [-1, 0, 1], shape=(N, N))



    # initial_X = np.arange(-M1, M2 + dx, dx)
    initial_V = np.full(N, S0 - K)
    V = initial_V

    # extra = np.zeros_like(V)
    # extra[-1] = np.exp(M2)

    for t in range(int(T / dt)):
        print(f"t = {t}", end="\r")
        V = sparse.linalg.inv(B) @ (A @ V)


    print(V)
    

finite_diff(100, 110, 0.04, 0.3, 1, 0.01, 0.01, 4, 4)