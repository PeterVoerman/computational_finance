def finite_diff_ftcs(S0, K, r, sigma, T, dx, dt, M1, M2):
    N = int(T / dt)
    NX = int((M2 + M1) / dx) + 2

    A_diag_1 = ((r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(NX - 1)
    A_diag_2 = (1 + 0.5 * sigma ** 2 * dt / dx ** 2 + 0.5 * r * dt) * np.ones(NX)
    A_diag_3 = (-(r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(NX - 1)

    B_diag_1 = (-(r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(NX- 1)
    B_diag_2 = (1 - 0.5 * sigma ** 2 * dt / dx ** 2 - 0.5 * r * dt) * np.ones(NX)
    B_diag_3 = ((r - 0.5 * sigma ** 2) * dt / (4 * dx ** 2) + 0.25 * sigma ** 2 * dt / dx ** 2) * np.ones(NX - 1)


    # Boundary conditions
    A_diag_1[0] = 0
    A_diag_3[-1] = 0

    B_diag_1[0] = 0
    B_diag_3[-1] = 0


    A = sparse.diags([A_diag_1, A_diag_2, A_diag_3], [-1, 0, 1], shape=(NX, NX))
    B = sparse.diags([B_diag_1, B_diag_2, B_diag_3], [-1, 0, 1], shape=(NX, NX))



    # initial_X = np.arange(-M1, M2 + dx, dx)
    initial_V = np.maximum(np.exp(np.linspace(-M1, M2, NX)) - K, 0)
    V = initial_V

    # extra = np.zeros_like(V)
    # extra[-1] = np.exp(M2)

    for t in range(int(T / dt)):
        print(f"t = {t}", end="\r")
        V = sparse.linalg.inv(B) @ (A @ V)


    print(V)
    

finite_diff_ftcs(100, 110, 0.04, 0.3, 1, 0.01, 0.01, 4, 4)