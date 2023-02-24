import numpy as np
import matplotlib.pyplot as plt

K = 50

Stlist = np.arange(30, 70, 0.5)
P1, P2 = [], []

for St in Stlist:
    P1.append(max(0, (St - K)) + K - K)
    P2.append(max(0, (K - St)) + St - K)

plt.title("Payoff diagram for two portfolios")
plt.plot(Stlist, P1, label = "call + cash", color = 'yellow', alpha = 0.5)
plt.plot(Stlist, P2, label = "put + stock", alpha = 0.5)
plt.axvline(K, ls='--', color = 'black')
plt.text(50.5, -.5, "K")

plt.xlabel("$S_T$")
plt.ylabel("Payoff")
plt.legend()

plt.show()