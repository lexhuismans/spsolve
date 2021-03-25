import solver
import database
import numpy as np

k_b = database.k_b

def test_fermi_dirac():
    E = np.linspace(-30, 30, 600)
    for E_f in np.linspace(-10, 10, 200):
        for T in np.linspace(0, 500, 500):
            assert np.all(solver.fermi_dirac(E_f, E, T) == 1 / (1 + np.exp((E - E_f) / k_b / T)))

            
def test_solve_poisson():
    assert True
    