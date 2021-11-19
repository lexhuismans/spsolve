from spsolve import solver, plotter, database
import numpy as np

layer1 = solver.Material("AlInSb", 200, 0.1)
layer2 = solver.Material("InSbAs", 30, .226)
layer3 = solver.Material("SiNx", 41)

stacked = solver.StackedLayers(0, 300, (False, 0), (True, -1.05555556), layer1, layer2, layer3)
print(stacked.m_c)
stacked.schrod_where = (150, 229)
stacked.CBO = -2.46890087

band, psi, energies, rho = stacked.solve_snider()
print(band)
layer1 = solver.Material("AlInSb", 100, 0.1)
layer2 = solver.Material("InSbAs", 30, .226)
layer3 = solver.Material("SiNx", 41)

stacked = solver.StackedLayers(0, 300, (False, 0), (True, -1.05555556), layer1, layer2, layer3)

stacked.schrod_where = (150, 229)
stacked.CBO = -2.46890087

plotter.plot_distributions(stacked.grid, band, psi, energies, rho)