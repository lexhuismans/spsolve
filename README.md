# spsolve
spsolve is a Python package that aims to solve the electrostatics in layered heterostructures through the coupled Schrödinger-Poisson equation. It has various features, including the k.p approximation and spin-orbit coupling. 

A simple example:
```python
from spsolve import solver
from spsolve.plotter import plot_distributions, plot_conduction_band, plot_poisson

# Material = namedtuple('Material', ['doping', 'm_eff', 'epsilon_r', 'band_offset' 'L'])
layer1 = solver.Material(0, 0.0929, 12.048, .2319, 10)
layer2 = solver.Material(0, 0.067, 12.9, 0, 15)
layer3 = solver.Material(0, 0.0929, 12.048, .2319, 10)

# StackedLayers(T, N, bound_left, bound_right, *layers)
stacked = solver.StackedLayers(300, 350, (True, 0.23), (False, 0), layer1, layer2, layer3)

stacked.solve_snider()

plot_distributions(stacked.grid, stacked.rho, stacked.band, stacked.transverse_modes, stacked.energies)
```