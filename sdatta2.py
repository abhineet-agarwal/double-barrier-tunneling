import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# Constants (all MKS, except energy which is in eV)
hbar = 1.06e-34  # Reduced Planck's constant (J·s)
q = 1.6e-19  # Elementary charge (C)
epsil = 10 * 8.85E-12  # Dielectric constant (F/m)
kT = 0.025  # Thermal energy at room temperature (eV)
m = 0.25 * 9.1e-31  # Effective mass of electron (kg)
n0 = 2 * m * kT * q / (2 * np.pi * hbar**2)  # Electron density parameter

# Inputs
a = 3e-10  # Lattice constant (m)
t = hbar**2 / (2 * m * a**2 * q)  # Hopping parameter (eV)
beta = q * a**2 / epsil  # Scaled inverse dielectric constant
Ns = 15  # Number of sites in source/drain regions
Nc = 70  # Number of sites in channel region
Np = Ns + Nc + Ns  # Total number of sites
XX = a * 1e9 * np.arange(1, Np + 1)  # Spatial grid (nm)
mu = 0.318  # Chemical potential (eV)
# Initial electron density calculation
def Fhalf(x):
    xx = np.linspace(0, abs(x) + 10, 251)
    dx = xx[1] - xx[0]
    fx = (2 * dx / np.sqrt(np.pi)) * np.sqrt(xx) / (1 + np.exp(xx - x))
    y = np.sum(fx)
    return y

Nd = 2*((n0/2)**1.5)*Fhalf(mu/kT)
Nd = Nd * np.concatenate([np.ones(Ns), 0.5 * np.ones(Nc), np.ones(Ns)])

#second derivative matrix for Poisson equation
D2 = -2 * np.diag(np.ones(Np)) + np.diag(np.ones(Np - 1), 1) + np.diag(np.ones(Np - 1), -1)
D2[0, 0] = -1
D2[-1, -1] = -1 #zero field condition

# Hamiltonian matrix
T = 2 * t * np.diag(np.ones(Np)) - t * np.diag(np.ones(Np - 1), 1) - t * np.diag(np.ones(Np - 1), -1)
# Current Operator
Jop = (q * t / ((Np - 1) * hbar)) * (1j) * (np.diag(np.ones(Np - 1), -1) - np.diag(np.ones(Np - 1), 1))
# Energy grid
NE = 301
E = np.linspace(-0.25, 0.5, NE)  # Energy grid (eV)
dE = E[1] - E[0]  # Energy step
zplus = 1e-12j  # Small imaginary part to avoid singularities
f0 = n0 * np.log(1 + np.exp((mu - E) / kT))  # Fermi-Dirac distribution

# Initial guess for potential U
U = np.concatenate([np.zeros(Ns), 0.2 * np.ones(Nc), np.zeros(Ns)])

#Voltage Bias Steps
NV = 5
VV = np.linspace(0, 0.25, NV)

for kV in range(NV):
    V = VV[kV]
    Fn = np.concatenate([
        mu * np.ones(Ns),
        (mu - 0.5 * V) * np.ones(Nc),
        (mu - V) * np.ones(Ns)
    ])
    f1 = n0 * np.log(1 + np.exp((mu - E) / kT))
    f2 = n0 * np.log(1 + np.exp((mu - V - E) / kT))
    ass=0
    i  = 10
    while i>0.8:
        sig1 = np.zeros((Np, Np), dtype=complex)
        sig2 = np.zeros((Np, Np), dtype=complex)
        rho=np.zeros(Np)
        for k in range(NE):
            ck = 1 - (E[k] + zplus - U[0]) / (2 * t)
            ka = np.arccos(ck)
            sig1[0, 0] = -t * np.exp(1j * ka)
            gam1 = 1j * (sig1 - sig1.conj().T)

            ck = 1 - (E[k] + zplus - U[Np-1]) / (2 * t)
            ka = np.arccos(ck)
            sig2[Np-1,Np -1] = -t * np.exp(1j * ka)
            gam2 = 1j * (sig2 - sig2.conj().T)
            G = inv((E[k] + zplus) * np.eye(Np) - T - np.diag(U) - sig1 - sig2)
            A1 = G.conj()*gam1*G
            A2 = G.conj()*gam2*G
            rho = rho + (dE*((f1[k]*A1)+(f2[k]*A2))/(2*np.pi))
        
            

        n = np.real(np.diag(rho))/a
        D=np.zeros(Np)
        for k in range(Np):
            z = (Fn[k] - U[k]) / kT
            D[k] = 2 * (n0 / 2)**1.5 * (Fhalf(z + 0.1) - Fhalf(z)) / (0.1 * kT)
        dN=n-Nd+((1/beta)*D2@U)
        
        dU=(-beta)*(inv(D2-(beta*np.diag(D))))@dN
        U=U + dU
        i = np.max(np.abs(dN)) / np.max(np.abs(Nd))
        ass=ass+1
        print(ass)
        print(i)


UU = np.zeros((Np, NV))
J = np.zeros((Np, NV))

for kV in range(NV):
    UU[:, kV] = U
    J[:, kV] = (-0.5 * q) * np.diag(rho @ Jop + Jop @ rho)

II = np.sum(J, axis=0)


np.savez('filename', UU=UU, J=J, II=II, VV=VV)

plt.plot(VV, II)
plt.xlabel('Voltage')
plt.ylabel('Total Current')
plt.show()