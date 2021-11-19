import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from spsolve import solver
from spsolve.plotter import plot_varying_gate, plot_distributions

plt.rcParams.update({'font.size': 15})

def sheet_charge(rho, dl):
    return np.sum(rho) * dl


V_gates = np.linspace(0, .1, 30)

V_surfs = np.linspace(-.15, 0, 3)

layer1 = solver.Material('AlGaAs', 10, .3)
layer2 = solver.Material("GaAs", 20)
layer3 = solver.Material('AlGaAs', 10, .3)

stacked = solver.StackedLayers(0, 300, (True, 0.1), (True, 0.1), layer1, layer2, layer3)
plot_varying_gate(stacked, V_gates, V_surfs, legend=False)

plt.plot(V_gates, -rho_sheet * 1e14)
plt.xlabel(r"$V_{Gate}$ (V)")
plt.ylabel(r"$n_{sheet}$ $q_e/cm^2$")
plt.tight_layout()
plt.show()
