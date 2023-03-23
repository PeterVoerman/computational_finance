import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def finite_diff_ftcs(S0, K, r, sigma, T, dx, dt, M1, M2):
    Nt = int(T / dt)
    Nx = int((M2 + M1) / dx) + 2

    A_diag_1 = (-(r - 0.5 * sigma ** 2) * dt / (2 * dx) + 0.5 * sigma ** 2 * dt / dx ** 2) * np.ones(Nx - 1)
    A_diag_2 = (1 - sigma ** 2 * dt / dx ** 2 - r * dt) * np.ones(Nx)
    A_diag_3 = ((r - 0.5 * sigma ** 2) * dt / (2 * dx) + 0.5 * sigma ** 2 * dt / dx ** 2) * np.ones(Nx - 1)

    A = sparse.diags([A_diag_1, A_diag_2, A_diag_3], [-1, 0, 1], format='csc')

    print(A.toarray())

    initial_V = np.full(Nx, max(S0 - K, 0))



S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
dx = 0.1
dt = 0.01
M1 = 10
M2 = 10

finite_diff_ftcs(S0, K, r, sigma, T, dx, dt, M1, M2)