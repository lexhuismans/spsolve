from spsolve import solver, database
import numpy as np
import pytest

k_b = database.k_b
epsilon_0 = database.epsilon_0

@pytest.fixture
def infinite_well():
    doping = 0
    m_eff = .067
    dielec_const = 12.9
    band_offset = .8
    L = 20
    material = solver.Material(0, .067, dielec_Const*epsilon_0, band_offset, L)

    T = 0
    N = 200
    return solver.StackedMaterials(T, N, (0, True), (0, True), material)

def test_fermi_dirac():
    E = np.linspace(-30, 30, 600)
    for E_f in np.linspace(-10, 10, 200):
        for T in np.linspace(0, 500, 500):
            assert np.all(solver.fermi_dirac(E_f, E, T) == 1 / (1 + np.exp((E - E_f) / k_b / T)))


def test_solve_poisson(infinite_well):
    assert True
