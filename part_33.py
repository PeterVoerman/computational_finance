from math import log, pi, sin, cos
from numpy import exp

S0 = 100
K = 99
N = 64
sigma = 0.5

T = 1
r = 0.06

# moet dit nou variabel ofzo? onduidelijk tussen opdracht en paper
t = 0

V_xt = 0

for k in range(N):
    a = log(S0 / K) + r * T - 12 * (sigma ** 2 * T) ** (1 / 2)
    b = log(S0 / K) + r * T + 12 * (sigma ** 2 * T) ** (1 / 2)

    # b - a, vaak nodig
    bma = 24 * (sigma ** 2 * T) ** (1 / 2)

    chi = lambda k: (1 + ((k * pi) / bma) ** 2) ** (-1) * (((-1) ** k) * exp(b) - cos(k * pi * a / bma) + ((k * pi) / bma) * sin(k * pi * a / bma))
    psi = lambda k: b if k == 0 else (bma / (k * pi) * sin(a / bma))

    Vk = lambda k: (2 / bma) * K * (chi(k) - psi(k))

    # characteristic function van y - x. Is dit de goede?
    phi = lambda u: exp(-(0.5 * (sigma ** 2) * t * (u ** 2)) + (r - 0.5 * sigma ** 2) * t * u * 1j)

    # wat is de ;x op pagina 5 van de paper?
    V_xt += (phi((k * pi) / bma) * exp(-k * pi * a * 1j / bma)).real * Vk(k)

print(V_xt)