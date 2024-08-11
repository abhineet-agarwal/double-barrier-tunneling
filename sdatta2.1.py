import numpy as np
import matplotlib.pyplot as plt

# Constants (all MKS, except energy which is in eV)
hbar = 1.06e-34
q = 1.6e-19
epsil = 10 * 8.85e-12
kT = 0.025
m = 0.25 * 9.1e-31
n0 = 2 * m * kT * q / (2 * np.pi * (hbar**2))

# Inputs
a = 3e-10
t = (hbar**2) / (2 * m * (a**2) * q)
beta = q * a * a / epsil
Ns = 15
Nc = 70
Np = Ns + Nc + Ns
XX = a * 1e9 * np.arange(1, Np + 1)
mu = 0.318
def Fhalf(x):
    xx = np.linspace(0, abs(x) + 10, 251)
    dx = xx[1] - xx[0]
    fx = (2 * dx / np.sqrt(np.pi)) * np.sqrt(xx) / (1 + np.exp(xx - x))
    y = np.sum(fx)
    return y

Nd = 2*((n0/2)**1.5)*Fhalf(mu/kT)
Nd = Nd * np.concatenate([np.ones(Ns), 0.5 * np.ones(Nc), np.ones(Ns)])

# d2/dx2 matrix for Poisson solution
D2 = -2 * np.diag(np.ones(Np)) + np.diag(np.ones(Np - 1), 1) + np.diag(np.ones(Np - 1), -1)
D2[0, 0] = -1
D2[-1, -1] = -1  # zero field condition

# Hamiltonian matrix
T = 2 * t * np.diag(np.ones(Np)) - t * np.diag(np.ones(Np - 1), 1) - t * np.diag(np.ones(Np - 1), -1)
Jop = (q * t / ((Np - 1) * hbar)) * 1j * (np.diag(np.ones(Np - 1), -1) - np.diag(np.ones(Np - 1), 1))

# Energy grid
NE = 301
E = np.linspace(-0.25, 0.5, NE)
dE = E[1] - E[0]
zplus = 1j * 1e-12
f0 = n0 * np.log(1 + np.exp((mu - E) / kT))

# Initial guess for U
U = np.concatenate((np.zeros(Ns), 0.2 * np.ones(Nc), np.zeros(Ns)))

# Voltage bias steps
NV = 5
VV = np.linspace(0, 0.25, NV)
UU = np.zeros((Np, NV))
J = np.zeros((Np, NV))

for kV in range(NV):
    V = VV[kV]
    Fn = np.concatenate((mu * np.ones(Ns), (mu - 0.5 * V) * np.ones(Nc), (mu - V) * np.ones(Ns)))
    f1 = n0 * np.log(1 + np.exp((mu - E) / kT))
    f2 = n0 * np.log(1 + np.exp((mu - V - E) / kT))

    convergence = 10
    while convergence > 5:
        sig1 = np.zeros((Np, Np), dtype=complex)
        sig2 = np.zeros((Np, Np), dtype=complex)
        rho = np.zeros((Np, Np), dtype=complex)

        for k in range(NE):
            ck1 = 1 - (E[k] + zplus - U[0]) / (2 * t)
            ka1 = np.arccos(ck1)
            sig1[0, 0] = -t * np.exp(1j * ka1)
            gam1 = 1j * (sig1 - np.conj(sig1.T))

            ck2 = 1 - (E[k] + zplus - U[-1]) / (2 * t)
            ka2 = np.arccos(ck2)
            sig2[-1, -1] = -t * np.exp(1j * ka2)
            gam2 = 1j * (sig2 - np.conj(sig2.T))

            G = np.linalg.inv((E[k] + zplus) * np.eye(Np) - T - np.diag(U) - sig1 - sig2)
            A1 = np.conj(G.T) @ gam1 @ G
            A2 = np.conj(G.T) @ gam2 @ G
            rho += dE * ((f1[k] * A1 + f2[k] * A2) / (2 * np.pi))

        n = (1 / a) * np.real(np.diag(rho))

        # Correction dU from Poisson
        D = np.zeros(Np)
        for k in range(Np):
            z = (Fn[k] - U[k]) / kT
            D[k] = 2 * ((n0 / 2)**1.5) * ((np.exp(z + 0.1) - np.exp(z)) / 0.1) / kT

        dN = n - Nd + (1 / beta) * D2 @ U
        dU = -beta * np.linalg.inv(D2 - beta * np.diag(D)) @ dN
        U += dU

        # Check for convergence
        convergence = np.max(np.abs(dN)) / np.max(Nd)
        print(convergence)
    UU[:, kV] = U
    J[:, kV] = -0.5 * q * np.diag(rho @ Jop + Jop @ rho)
    
II = np.sum(J, axis=0)

# Plot the results
plt.figure()
plt.plot(VV, II)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('IV Characteristics')
plt.grid(True)
plt.show()
