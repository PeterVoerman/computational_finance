from math import log, pi, sin, cos
from numpy import exp

S0 = 120
K = 110
N = 64
sigma = 0.3

T = 1
r = 0.04

t = 0

a = log(S0 / K) + r * T - 12 * (sigma ** 2 * T) ** (1 / 2)
b = log(S0 / K) + r * T + 12 * (sigma ** 2 * T) ** (1 / 2)

# b - a, vaak nodig
bma = 24 * (sigma ** 2 * T) ** (1 / 2)

chi = lambda k: (1 + ((k * pi) / bma) ** 2) ** (-1) * (((-1) ** k) * exp(b) - cos(k * pi * a / bma) + ((k * pi) / bma) * sin(k * pi * a / bma))
psi = lambda k: b if k == 0 else (bma / (k * pi)) * sin(k * pi * a / bma)

Vk = lambda k: (2 / bma) * K * (chi(k) - psi(k))

# characteristic function van y - x. Is dit de goede?
# phi = lambda u: exp(-(0.5 * (sigma ** 2) * t * (u ** 2)) + u * (r - 0.5 * sigma ** 2) * t * 1j)
# Fk = lambda k: (2 / bma) * (phi((k * pi) / bma) * exp((-k * a * pi * 1j) / bma)).real

phi = lambda u: exp(1j * u * (r - 0.5 * sigma ** 2) * t - (0.5 * sigma ** 2 * t * u ** 2))
Fk = lambda k: (2 / bma) * (phi(k * pi / bma) * exp(-1j * (k * a * pi) / (bma))).real

# goede waarden voor Fk
# k = 0: 0.277777777
# k = 1: 0.0054
# k = 10: -0.1156

# print(Fk(1))

V_xt = 0

for k1 in range(N + 1):
    u = (k1 * pi) / bma
    print(u)
    Fk2 = (2 / bma) * exp(a * (u ** 2) * (r - 0.5 * (sigma ** 2) * t)) * cos(0.5 * (sigma ** 2) * t * a * (u ** 3))
    print(k1, Fk2, Fk(k1))
    V_xt += Fk2 * Vk(k1)

V_xt *= exp(-r * t) * bma / 2
print(V_xt)