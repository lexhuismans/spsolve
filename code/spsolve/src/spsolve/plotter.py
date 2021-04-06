import math
import time

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import root as scipy_root

from . import solver

# Physical constants
h_bar = 0.276042828  # eV s
m_eff = 1.08  # effective mass of electron
PERMETTIVITY = 0.055263494  # q_e/(V*nm)
q_e = 1  # elementary charge


def plot_charge(grid, rho, ax=None, rho_fit=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, rho, label="Numerical")
    if not isinstance(rho_fit, type(None)):
        ax.plot(grid[1:-1], rho_fit[1:-1], label="Analytical")
        ax.legend()

    ax.set_xlabel("Position (nm)")
    ax.set_ylabel(r"Charge ($q_e/nm^3$)")

    if ax is None:
        plt.show()
    else:
        ax.set_title("Charge distribution")
        return ax


def plot_band(grid, band, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, band)

    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("Energy (eV)")

    if ax is None:
        plt.show()
    else:
        ax.set_title("Conduction band")
        return ax


def plot_wave(grid, psi, energies, ax=None, n_waves = 3):
    if ax is None:
        fig, ax = plt.subplots(1)

    ax.plot(grid, psi[:, 0] ** 2, label="$E_0$ = {:f} eV".format(energies[0]))
    for n in np.arange(n_waves - 1) + 1:
        ax.plot(grid, psi[:, n] ** 2, label="$E_{}$ = {:f} eV".format(n, energies[n]))

    ax.set_xlabel("Position (nm)")
    ax.set_ylabel("$|\psi|^2$")

    if ax is None:
        plt.show()
    else:
        ax.set_title("Probability ($nm^{-1}$)")
        return ax


def plot_distributions(grid, rho, band, psi, energies):
    fig, ax = plt.subplots(3)

    plot_charge(grid, rho, ax[0])
    plot_band(grid, band, ax[1])
    plot_wave(grid, psi, energies, ax = ax[2])

    fig.tight_layout()
    plt.show()


def plot_charge_density(startV=-1, stopV=1):
    N = 200
    layer = solver.Material('GaAs', 100)
    syst = solver.StackedLayers(0, N, (True, startV), (True, startV), layer)

    N_V = 100

    rho_mid = np.zeros(N_V)

    V = np.linspace(startV, stopV, N_V)

    for i in range(N_V):
        V_0 = np.ones(N)*V[i]
        band = -q_e*V_0
        trans_modes, energies = syst.solve_schrodinger(band)

        rho = syst.solve_charge(trans_modes, energies)
        rho_mid[i] = rho[int(N / 2)]

    plt.plot(V, rho_mid)
    plt.xlabel("$V_0$ (V)")
    plt.ylabel("$n_e$ ($q_e/nm^3$)")
    plt.show()


def plot_E_n():
    # System
    l = 20  # length of the system (nm)
    N = 500  # number of gridpoints
    N_V = 40
    grid = np.linspace(0, l, N)
    dl = grid[1]
    E_fermi = 0.8  # eV

    # Boundary conditions
    dirichlet_left = True
    potential_left_boundary = 0
    dirichlet_right = False
    potential_right_boundary = 0

    potential = np.zeros(N)
    psi = np.zeros(N)

    # Poisson matrix
    pois_matrix = solver.comp_pois_matrix(dl, N, dirichlet_left, dirichlet_right)
    syst = solver.make_system(dl, N)

    n_e_mid = np.zeros(N_V)
    V = np.linspace(-0.799, -0.801, N_V)
    E_n_max = 10
    E_n = np.zeros((E_n_max, N_V))

    for i in range(N_V):
        V_0 = V[i]
        n_e = np.zeros(N)
        # Solve for potential with Poisson equation
        phi = solver.solve_poisson(pois_matrix, n_e, dl, V_0, potential_right_boundary)

        # Solve for wavefunction with Schrodinger equation
        energies, psi = solver.solve_syst(syst, phi, dl)
        E_n[:, i] = energies[:E_n_max]

    for j in range(E_n_max):
        plt.plot(V, E_n[j, :], label="n = {}".format(j))

    z = np.polyfit(V, E_n[0, :], 1)
    y = z[1] + z[0] * V
    error = np.sum((y - E_n[0, :]) ** 2) / len(V)
    print("Fit error: ", error)

    plt.plot(V, y, label="Fit")
    plt.xlabel("$V_0$")
    plt.ylabel("$E_n$ (eV)")
    plt.legend()
    plt.savefig("E_n.png")
