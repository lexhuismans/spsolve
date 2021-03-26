import math

import numpy as np
import matplotlib.pyplot as plt
import pytest
from spsolve import database, solver

k_b = database.k_b
h_bar = database.h_bar
epsilon_0 = database.epsilon_0
q_e = database.q_e

@pytest.fixture
def infinite_well():
    doping = 0
    m_eff = 0.067
    dielec_const = 12.9
    band_offset = 0.8
    L = 20
    material = solver.Material(doping, m_eff, dielec_const * epsilon_0, band_offset, L)

    T = 0
    N = 400
    return solver.StackedLayers(T, N, (True, -.05), (True, -.05), material)


def test_fermi_dirac():
    E = np.linspace(-30, 30, 600)
    for E_f in np.linspace(-10, 10, 200):
        for T in np.linspace(0, 500, 500):
            assert np.all(solver.fermi_dirac(E_f, E, T) == 1 / (1 + np.exp((E - E_f) / k_b / T)))


def test_solve_schrodinger(infinite_well):
    grid = infinite_well.grid
    L = infinite_well.L
    N = infinite_well.N
    psi = np.zeros((N, N))
    energies = np.zeros(N)

    for V_0 in np.linspace(-2, 2, 20):
        phi = np.ones(N)*V_0

        for n in np.arange(N) + 1:
            psi[:, n - 1] = math.sqrt(2 / L) * np.sin(grid * n * math.pi / L)
            energies[n - 1] = (n * math.pi * h_bar) ** 2 / (2 * infinite_well.m_eff[0] * L ** 2) - V_0

        psi_test, energies_test = infinite_well.solve_schrodinger(phi)

        assert np.all(np.abs(psi[:,0:10]) == pytest.approx(np.abs(psi_test[:,0:10]), 0.01))
        assert np.all(energies[0:10] == pytest.approx(energies_test[0:10], 0.1))

def test_solve_poisson(infinite_well):
    pass

def test_solve_charge(infinite_well):
    grid = infinite_well.grid
    L = infinite_well.L
    N = infinite_well.N
    m_eff = infinite_well.m_eff[0]
    psi = np.zeros((N, N))
    energies = np.zeros(N)
    rho = np.zeros(N)
    V_0 = -0.05
    phi = np.ones(N)*V_0 # Potential

    for n in np.arange(N) + 1:
        psi[:, n - 1] = math.sqrt(2 / L) * np.sin(grid * n * math.pi / L)
        energies[n - 1] = (n * math.pi * h_bar) ** 2 / (2 * m_eff * L ** 2) - V_0

    inner_product = psi**2
    rho = -q_e * m_eff/(math.pi * h_bar**2) * np.dot(inner_product, -energies*(energies < 0))

    assert np.all(rho == infinite_well.solve_charge(psi, energies))
